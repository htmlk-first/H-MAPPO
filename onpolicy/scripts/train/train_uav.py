import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
# $ python train_uav.py --env_name UAV --algorithm_name rmappo --experiment_name uav_test --num_agents 4 --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_env_steps 200000 --ppo_epoch 10 --episode_length 200

# MAPPO 코드를 import 하기 위한 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.uav.uav_env import UAVEnv # 우리가 만든 환경을 import
from onpolicy.runner.shared.uav_runner import UAVRunner as Runner # 우리가 만들 러너를 import

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "UAV":
                env = UAVEnv()
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # env.seed()는 gymnasium 최신 버전에서 권장되지 않으므로 삭제하거나 주석 처리합니다.
            # env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    # envs를 함수가 아닌, 래핑된 객체로 반환 ---
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        # 다중 스레드를 위한 코드 (향후 사용 가능)
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    # 우리 환경에 맞는 파라미터들을 추가
    parser.add_argument('--num_agents', type=int, default=4, help="number of agents")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    all_args.use_wandb = False # wandb on/off

    # cuda, wandb, process title 등 학습 환경 설정
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 결과 저장을 위한 디렉토리 설정
    run_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")) \
              / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env setup
    envs = make_train_env(all_args)
    eval_envs = None # 평가 환경은 일단 생략

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = Runner(config)
    runner.run()
    
    # post-processing
    envs.close()
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])