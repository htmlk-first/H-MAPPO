import sys
import os
import numpy as np
import torch
import imageio
from pathlib import Path
import copy

# MAPPO 코드를 import 하기 위한 경로 설정
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from onpolicy.config import get_config
from onpolicy.envs.uav.uav_env import UAVEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

# H-MAPPO 모델을 불러오기 위해 R_Actor 대신 R_MAPPOPolicy를 임포트합니다.
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from onpolicy.algorithms.utils.util import check

def _t2n(x):
    """토치 텐서를 넘파이 배열로 변환합니다."""
    return x.detach().cpu().numpy()

# train_uav.py의 make_train_env를 기반으로 렌더링 환경 생성 함수를 만듭니다.
def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # UAVEnv에 all_args를 전달합니다.
            env = UAVEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    # 렌더링은 스레드 1개로 DummyVecEnv를 사용합니다.
    return DummyVecEnv([get_env_fn(0)])

# train_uav.py에서 UAV 환경 인자 파싱 로직을 가져옵니다.
def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="simple_uav", help="which scenario to run on.")
    parser.add_argument("--high_level_timestep", type=int, default=15, help="Frequency of high-level policy action.")
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
    # parse_args 함수를 호출합니다.
    all_args = parse_args(args, parser)

    # --- 렌더링을 위한 파라미터 설정 ---
    all_args.n_rollout_threads = 1
    # num_agents를 num_uavs로 설정합니다.
    all_args.num_agents = all_args.num_uavs
    all_args.use_render = True
    all_args.save_gifs = True
    all_args.ifi = 0.1
    all_args.use_recurrent_policy = True # H-MAPPO는 RNN을 사용합니다.
    # ----------------------------------

    # cuda 설정
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

    # env setup
    # make_render_env 함수를 사용합니다.
    envs = make_render_env(all_args)
    
    # --- H-MAPPO 정책 네트워크 설정 (train_uav.py 로직) ---

    # 1. Low-level Policy
    low_level_obs_space = envs.observation_space['low_level'][0]
    low_level_share_obs_space = envs.share_observation_space['low_level'][0]
    low_level_act_space = envs.action_space['low_level'][0]
    
    low_level_policy = Policy(all_args,
                              low_level_obs_space,
                              low_level_share_obs_space,
                              low_level_act_space,
                              device=device)

    # 2. High-level Policy
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
    # ----------------------------------------------------
    
    # H-MAPPO 모델 로드
    print(f"Loading models from {all_args.model_dir}")
    try:
        # H_UAVRunner가 high_level_actor.pt, low_level_actor.pt로 저장했다고 가정합니다.
        low_policy_state_dict = torch.load(str(all_args.model_dir) + '/low_level_actor.pt', map_location=torch.device('cpu'))
        low_level_policy.actor.load_state_dict(low_policy_state_dict)
        
        high_policy_state_dict = torch.load(str(all_args.model_dir) + '/high_level_actor.pt', map_location=torch.device('cpu'))
        high_level_policy.actor.load_state_dict(high_policy_state_dict)
        
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {all_args.model_dir}에 'low_level_actor.pt'와 'high_level_actor.pt' 파일이 있는지 확인하세요.")
        print(f"(만약 H_UAVRunner.py에 save 로직을 추가하지 않았다면 모델 파일이 생성되지 않습니다.)")
        print(f"Original error: {e}")
        return
        
    low_level_policy.actor.eval()
    high_level_policy.actor.eval()

    frames = []
    # DummyVecEnv.reset()은 (obs, info) 튜플을 반환합니다.
    full_obs_tuple = envs.reset()
    full_obs = full_obs_tuple[0] # obs는 튜플의 첫 번째 요소입니다.
    
    # H-MAPPO를 위한 RNN 상태 초기화
    # Low-level: (n_threads, n_agents, n_layers, hidden_size) -> (1, 4, 1, 64)
    low_rnn_states = np.zeros((1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    # High-level: (n_threads, 1, n_layers, hidden_size) -> (1, 1, 1, 64)
    high_rnn_states = np.zeros((1, 1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    
    # Masks
    masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)
    high_masks = np.ones((1, 1, 1), dtype=np.float32)


    for step in range(all_args.episode_length):
        if all_args.use_render:
            # DummyVecEnv의 render는 프레임 리스트를 반환합니다.
            frame = envs.render(mode='rgb_array')[0]
            if all_args.save_gifs:
                frames.append(frame)

        with torch.no_grad():
            
            # --- 1. 상위 레벨 행동 (주기적) ---
            if step % all_args.high_level_timestep == 0:
                # high_obs shape: (1, 1, obs_dim)
                high_obs_input = full_obs[0]['high_level']
                # rnn_state shape: (1*1, 1, 64) -> (1, 64)
                high_rnn_states_input = high_rnn_states.reshape(-1, all_args.hidden_size)
                high_masks_input = high_masks.reshape(-1, 1)

                high_actions, high_rnn_states_out = high_level_policy.act(
                    obs=check(np.array(high_obs_input)).float(),
                    rnn_states_actor=check(high_rnn_states_input).float(),
                    masks=check(high_masks_input).float(),
                    deterministic=True
                )
                high_actions_np = _t2n(high_actions)
                high_rnn_states = _t2n(high_rnn_states_out).reshape(1, 1, all_args.recurrent_N, all_args.hidden_size)
                
                # 환경에 목표(Goal) 설정
                envs.env_method('set_goals', high_actions_np)

            # --- 2. 하위 레벨 행동 (매 스텝) ---
            # low_obs shape: (1, 4, obs_dim)
            low_obs_input = full_obs[0]['low_level']
            # rnn_state shape: (1*4, 1, 64) -> (4, 64)
            low_rnn_states_input = low_rnn_states.reshape(-1, all_args.hidden_size)
            low_masks_input = masks.reshape(-1, 1)
            
            low_actions, low_rnn_states_out = low_level_policy.act(
                obs=check(np.array(low_obs_input).reshape(all_args.num_agents, -1)).float(),
                rnn_states_actor=check(low_rnn_states_input).float(),
                masks=check(low_masks_input).float(),
                deterministic=True
            )
            low_actions_np = _t2n(low_actions)
            low_rnn_states = _t2n(low_rnn_states_out).reshape(1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size)

        # envs.step()은 (n_threads, n_agents, act_dim) 형태의 입력을 기대합니다.
        full_obs, rewards, dones, infos = envs.step([low_actions_np]) # [low_actions_np]는 (1, 4, act_dim)

        dones_env = dones
        if np.all(dones_env):
            print("Episode Finished.")
            break
        
        # 종료 시 RNN 상태 초기화
        low_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        high_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), 1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        masks[dones_env == True] = np.float32(0.0)
        high_masks[dones_env == True] = np.float32(0.0)
            
    if all_args.save_gifs:
        # GIF 저장 경로를 model_dir의 부모(runXX) 디렉토리 밑으로 변경
        render_dir = Path(str(all_args.model_dir)).parent / "renders"
        render_dir.mkdir(exist_ok=True)
        gif_path = render_dir / 'render.gif'
        print(f"Saving GIF to {gif_path}")
        imageio.mimsave(gif_path, frames, duration=all_args.ifi, loop=0)

    envs.close()

if __name__ == '__main__':
    main(sys.argv[1:])