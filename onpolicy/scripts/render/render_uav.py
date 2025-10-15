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
from onpolicy.envs.uav.parameters import SIM_TIME_STEPS

def main():
    parser = get_config()
    all_args = parser.parse_args()

    # --- 렌더링을 위한 파라미터 설정 ---
    all_args.n_rollout_threads = 1
    all_args.num_agents = 4
    all_args.episode_length = SIM_TIME_STEPS
    all_args.use_render = True
    all_args.save_gifs = True
    all_args.ifi = 0.1
    # ----------------------------------

    # env setup
    env = UAVEnv()
    
    # actor network
    actor = R_Actor(all_args, env.observation_space[0], env.action_space[0])
    
    # load model
    print(f"Loading model from {all_args.model_dir}")
    try:
        actor_state_dict = torch.load(str(all_args.model_dir) + '/actor.pt', map_location=torch.device('cpu'))
        actor.load_state_dict(actor_state_dict)
    except FileNotFoundError:
        print(f"Error: Model file not found at {all_args.model_dir}. Please train a model first.")
        return
        
    actor.eval()

    frames = []
    obs, info = env.reset()
    
    # rnn_states를 NumPy 배열로 초기화
    rnn_states = np.zeros((all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((all_args.num_agents, 1), dtype=np.float32)

    for step in range(all_args.episode_length):
        if all_args.use_render:
            frame = env.render(mode='rgb_array')
            if all_args.save_gifs:
                frames.append(frame)

        with torch.no_grad():
            # 버퍼에서 오는 데이터와 모양을 맞추기 위해 rnn_states를 3D -> 2D로 변경
            rnn_states_input = rnn_states.reshape(-1, all_args.hidden_size)
            
            action_tensor, _, rnn_states_tensor = actor(torch.from_numpy(np.array(obs)).float(), 
                                                        torch.from_numpy(rnn_states_input).float(), 
                                                        torch.from_numpy(masks).float(), 
                                                        deterministic=True)
            
            # 신경망의 출력(Tensor)을 다음 루프를 위해 다시 NumPy 배열로 변환
            rnn_states = rnn_states_tensor.detach().cpu().numpy().reshape(all_args.num_agents, all_args.recurrent_N, all_args.hidden_size)                
            # --------------------
        actions = np.array(np.split(action_tensor.detach().cpu().numpy(), all_args.n_rollout_threads))
        obs, rewards, dones, infos = env.step(actions[0])

        if np.all(dones):
            print("Mission accomplished!")
            break
            
    if all_args.save_gifs:
        render_dir = Path(str(all_args.model_dir) + "/renders")
        render_dir.mkdir(exist_ok=True)
        gif_path = render_dir / 'render.gif'
        print(f"Saving GIF to {gif_path}")
        imageio.mimsave(gif_path, frames, duration=all_args.ifi, loop=0)

    env.close()

if __name__ == '__main__':
    main()