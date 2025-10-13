import sys
import os
import numpy as np
import torch
import imageio
from pathlib import Path

# MAPPO 코드를 import 하기 위한 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from onpolicy.config import get_config
from onpolicy.envs.uav.uav_env import UAVEnv
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

def main():
    parser = get_config()
    all_args = parser.parse_args()

    # --- 추가된/수정된 파라미터 ---
    all_args.n_rollout_threads = 1
    all_args.num_agents = 4 # UAV 수
    all_args.episode_length = 500 # 시뮬레이션 최대 길이
    all_args.use_render = True # 렌더링 사용 여부
    all_args.save_gifs = True # GIF 저장 여부
    # ----------------------------

    # env setup
    env = UAVEnv()
    
    # actor network
    actor = R_Actor(all_args, env.observation_space[0], env.action_space[0])
    
    # load model
    print(f"Loading model from {all_args.model_dir}")
    actor_state_dict = torch.load(str(all_args.model_dir) + '/actor.pt', map_location=torch.device('cpu'))
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    # GIF 저장을 위한 준비
    frames = []
    
    # Run a single episode
    obs, info = env.reset()
    
    rnn_states = np.zeros((all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((all_args.num_agents, 1), dtype=np.float32)

    for step in range(all_args.episode_length):
        if all_args.use_render:
            frame = env.render(mode='rgb_array')
            if all_args.save_gifs:
                frames.append(frame)

        # Get actions from the loaded policy
        with torch.no_grad():
            action, _, rnn_states = actor(torch.from_numpy(np.array(obs)), 
                                        torch.from_numpy(rnn_states), 
                                        torch.from_numpy(masks), 
                                        deterministic=True)
        
        # Unpack actions and step the environment
        actions = np.array(np.split(action.detach().cpu().numpy(), all_args.n_rollout_threads))
        obs, rewards, dones, infos = env.step(actions[0])

        if np.all(dones):
            break

    # Save GIF
    if all_args.save_gifs:
        gif_path = str(all_args.model_dir) + '/render.gif'
        print(f"Saving GIF to {gif_path}")
        imageio.mimsave(gif_path, frames, duration=0.1, loop=0)

    env.close()

if __name__ == '__main__':
    main()