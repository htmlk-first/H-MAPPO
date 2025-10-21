import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import copy

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from onpolicy.config import get_config
from onpolicy.envs.uav.uav_env import UAVEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # args를 전달하여 환경을 생성합니다.
            env = UAVEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    
def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = UAVEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="simple_uav", help="which scenario to run on.")
    # [추가] H-MAPPO를 위한 새로운 파라미터들
    parser.add_argument("--high_level_timestep", type=int, default=15, help="Frequency of high-level policy action.")
    
    # [추가] UAV 환경을 위한 파라미터들 (기존 parameters.py 내용을 여기에 통합)
    parser.add_argument("--num_uavs", type=int, default=4, help="number of UAVs")
    parser.add_argument("--num_pois", type=int, default=12, help="number of Points of Interest")
    parser.add_argument("--num_jammers", type=int, default=2, help="number of Jammers")
    parser.add_argument("--world_size", type=float, default=100.0, help="size of the world")
    parser.add_argument("--fly_th", type=float, default=10.0, help="flying altitude of UAVs")
    parser.add_argument("--v_max", type=float, default=5.0, help="maximum velocity of UAVs")
    parser.add_argument("--u_max", type=float, default=2.0, help="maximum acceleration of UAVs")
    parser.add_argument("--jammer_radius", type=float, default=15.0, help="radius of jammer interference")
    parser.add_argument("--jammer_power", type=float, default=1.0, help="power of jammer signal")
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.num_agents = all_args.num_uavs  # 에이전트 수를 UAV 수로 설정

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    # assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy or all_args.use_centralized_v) == False, ("check args!")

    # cuda
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

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
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

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    # config 딕셔너리 생성
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
    }

    # Runner를 H_UAVRunner로 변경하고, 계층적 정책을 생성
    from onpolicy.runner.shared.h_uav_runner import H_UAVRunner as Runner
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
    from onpolicy.utils.shared_buffer import SharedReplayBuffer as ReplayBuffer

    # 1. Low-level Policy and Trainer
    # 환경으로부터 하위 레벨의 관측/행동 공간 정보를 가져옴
    low_level_obs_space = envs.observation_space['low_level'][0]
    low_level_share_obs_space = envs.share_observation_space['low_level'][0]
    low_level_act_space = envs.action_space['low_level'][0]
    
    low_level_policy = Policy(all_args,
                              low_level_obs_space,
                              low_level_share_obs_space,
                              low_level_act_space,
                              device=device)
    
    low_level_trainer = TrainAlgo(all_args, low_level_policy, device=device)
    
    low_level_buffer = ReplayBuffer(all_args,
                                    all_args.num_uavs,
                                    low_level_obs_space,
                                    low_level_share_obs_space,
                                    low_level_act_space)

    # 2. High-level Policy and Trainer
    # 상위 레벨의 config를 별도로 생성
    high_level_args = copy.deepcopy(all_args)
    high_level_args.num_agents = 1 # 상위 에이전트는 1명
    
    high_level_obs_space = envs.observation_space['high_level'][0]
    high_level_share_obs_space = envs.share_observation_space['high_level'][0]
    high_level_act_space = envs.action_space['high_level'][0]

    high_level_policy = Policy(high_level_args,
                               high_level_obs_space,
                               high_level_share_obs_space,
                               high_level_act_space,
                               device=device)
    
    high_level_trainer = TrainAlgo(high_level_args, high_level_policy, device=device)
    
    high_level_buffer = ReplayBuffer(high_level_args,
                                     1, # 상위 에이전트는 1명
                                     high_level_obs_space,
                                     high_level_share_obs_space,
                                     high_level_act_space)

    # Runner에 두 레벨의 trainer와 buffer를 리스트로 전달
    config["trainer"] = [high_level_trainer, low_level_trainer]
    config["buffer"] = [high_level_buffer, low_level_buffer]
    
    config["num_agents"] = all_args.num_uavs
    config["run_dir"] = run_dir
    
    runner = Runner(config)
    runner.run()
    
    # post-processing
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])