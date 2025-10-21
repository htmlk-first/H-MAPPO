import time
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class H_UAVRunner(Runner):
    def __init__(self, config):
        # BaseRunner의 필수 설정들을 직접 가져와 초기화
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.hidden_size = self.all_args.hidden_size
        self.recurrent_N = self.all_args.recurrent_N
        self.use_wandb = self.all_args.use_wandb
        self.log_interval = self.all_args.log_interval
        self.save_interval = self.all_args.save_interval

        # dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # H-MAPPO를 위한 trainer와 buffer 리스트를 직접 할당
        self.trainer = config['trainer']
        self.buffer = config['buffer']
        
        self.high_level_trainer = self.trainer[0]
        self.high_level_buffer = self.buffer[0]
        self.low_level_trainer = self.trainer[1]
        self.low_level_buffer = self.buffer[1]

        self.high_level_timestep = self.all_args.high_level_timestep

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.all_args.use_linear_lr_decay:
                self.high_level_trainer.policy.lr_decay(episode, episodes)
                self.low_level_trainer.policy.lr_decay(episode, episodes)

            # 에피소드 루프 시작
            for step in range(self.episode_length):
                # Sample actions
                low_values, low_actions, low_action_log_probs, low_rnn_states, low_rnn_states_critic = self.collect_low_level()

                high_actions = None
                if step % self.high_level_timestep == 0:
                    high_values, high_actions, high_action_log_probs, high_rnn_states, high_rnn_states_critic = self.collect_high_level()
                    self.envs.env_method('set_goals', high_actions)
                
                # Observe reward and next obs
                full_obs, rewards, dones, infos = self.envs.step(low_actions)

                # Unpack and restructure obs
                obs, share_obs = self.unpack_obs(full_obs)
                
                low_data = obs, share_obs, rewards, dones, infos, low_values, low_actions, low_action_log_probs, low_rnn_states, low_rnn_states_critic
                self.insert_low_level(low_data)

                if step % self.high_level_timestep == 0:
                    high_data = obs, share_obs, rewards, dones, infos, high_values, high_actions, high_action_log_probs, high_rnn_states, high_rnn_states_critic
                    self.insert_high_level(high_data)

            self.compute_and_train()
            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                
            if (episode % self.log_interval == 0):
                end = time.time()
                print(f"\n Episode {episode}/{episodes}, total num timesteps {total_num_steps} || FPS {int(total_num_steps / (end - start))}")

    def warmup(self):
        # DummyVecEnv.reset()은 (obs, infos) 튜플을 반환합니다.
        full_obs_tuple = self.envs.reset()
        # obs, share_obs를 얻기 위해 튜플의 첫 번째 요소(obs)만 전달합니다.
        obs, share_obs = self.unpack_obs(full_obs_tuple[0])
        
        self.low_level_buffer.share_obs[0] = share_obs['low_level'].copy()
        self.low_level_buffer.obs[0] = obs['low_level'].copy()

        self.high_level_buffer.share_obs[0] = share_obs['high_level'].copy()
        self.high_level_buffer.obs[0] = obs['high_level'].copy()
        
    def unpack_obs(self, full_obs_stacked):
        # full_obs_stacked는 (n_rollout_threads, ) 형태이며, 
        # 각 요소는 {'high_level': ..., 'low_level': ...} 딕셔너리입니다.
        # (n_rollout_threads=1일 때) full_obs_stacked.shape = (1,)
        
        # 1. High-level obs/share_obs (둘은 동일하며, 이미 중앙 집중형임)
        # d['high_level']의 shape: (1, high_obs_dim)
        # high_level_obs의 shape: (n_rollout_threads, 1, high_obs_dim)
        high_level_obs = np.array([d['high_level'] for d in full_obs_stacked])

        # 2. Low-level obs (개별 에이전트 관측)
        # d['low_level']의 shape: (num_agents, low_obs_dim)
        # obs_low_level_stacked의 shape: (n_rollout_threads, num_agents, low_obs_dim)
        obs_low_level_stacked = np.array([d['low_level'] for d in full_obs_stacked])

        # 3. Low-level share_obs (모든 에이전트의 관측을 하나로 합친 중앙 집중형 관측)
        # uav_env.py에 정의된 share_low_obs_dim = low_obs_dim * num_uavs
        
        n_rollout_threads = len(full_obs_stacked)
        
        # (n_threads, num_agents, low_obs_dim) -> (n_threads, num_agents * low_obs_dim)
        share_obs_low_level_flat = obs_low_level_stacked.reshape(n_rollout_threads, -1)
        
        # (n_threads, num_agents * low_obs_dim) -> (n_threads, 1, num_agents * low_obs_dim)
        share_obs_low_level_expanded = np.expand_dims(share_obs_low_level_flat, 1)
        
        # (n_threads, 1, ...) -> (n_threads, num_agents, ...)
        # 각 에이전트가 동일한 중앙 집중형 관측을 공유하도록 복제(tile)합니다.
        share_obs_low_level_tiled = np.tile(
            share_obs_low_level_expanded, 
            (1, self.num_agents, 1)
        )

        # 최종 obs/share_obs 딕셔너리 반환
        obs = {
            'low_level': obs_low_level_stacked,
            'high_level': high_level_obs
        }
        share_obs = {
            'low_level': share_obs_low_level_tiled,
            'high_level': high_level_obs # High-level obs는 이미 중앙 집중형이므로 그대로 사용
        }
        return obs, share_obs

    @torch.no_grad()
    def collect_low_level(self):
        self.low_level_trainer.prep_rollout()
        # (n_threads, n_agents, dim) -> (n_threads * n_agents, dim)으로 reshape
        buffer = self.low_level_buffer
        step = buffer.step
        
        share_obs_input = buffer.share_obs[step].reshape(-1, buffer.share_obs.shape[-1])
        obs_input = buffer.obs[step].reshape(-1, buffer.obs.shape[-1])
        # rnn_states도 (n_threads * n_agents, recurrent_N * hidden_size) 또는 (n_threads * n_agents, hidden_size)로 reshape
        # rnn.py가 (B*N, hidden_size)를 예상하므로 rnn_states.shape[-1]이 hidden_size여야 함 (recurrent_N=1일 때)
        rnn_states_input = buffer.rnn_states[step].reshape(-1, buffer.rnn_states.shape[-1])
        rnn_states_critic_input = buffer.rnn_states_critic[step].reshape(-1, buffer.rnn_states_critic.shape[-1])
        masks_input = buffer.masks[step].reshape(-1, buffer.masks.shape[-1])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.low_level_trainer.policy.get_actions(share_obs_input,
                                                        obs_input,
                                                        rnn_states_input,
                                                        rnn_states_critic_input,
                                                        masks_input)
            
        # 정책 출력을 (n_threads, n_agents, dim) 형태로 재구성
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def collect_high_level(self):
        self.high_level_trainer.prep_rollout()
        # (n_threads, 1, dim) -> (n_threads, dim)으로 reshape
        buffer = self.high_level_buffer
        step = buffer.step

        share_obs_input = buffer.share_obs[step].reshape(-1, buffer.share_obs.shape[-1])
        obs_input = buffer.obs[step].reshape(-1, buffer.obs.shape[-1])
        rnn_states_input = buffer.rnn_states[step].reshape(-1, buffer.rnn_states.shape[-1])
        rnn_states_critic_input = buffer.rnn_states_critic[step].reshape(-1, buffer.rnn_states_critic.shape[-1])
        masks_input = buffer.masks[step].reshape(-1, buffer.masks.shape[-1])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.high_level_trainer.policy.get_actions(share_obs_input,
                                                         obs_input,
                                                         rnn_states_input,
                                                         rnn_states_critic_input,
                                                         masks_input)
            
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert_low_level(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        # rnn_states shape: (n_threads, n_agents, hidden_size)
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.float32(0.0)

        # low_level 보상만 추출하여 (n_threads, n_agents, 1) 모양으로 변환합니다.
        low_rewards_list = [r['low_level'] for r in rewards]
        low_rewards_array = np.array(low_rewards_list) # shape (n_threads, n_agents)
        low_rewards_reshaped = np.expand_dims(low_rewards_array, axis=-1) # shape (n_threads, n_agents, 1)

        # rnn_states의 shape (1, 4, 64) -> (1, 4, 1, 64)로 변경
        rnn_states_expanded = np.expand_dims(rnn_states, axis=2)
        rnn_states_critic_expanded = np.expand_dims(rnn_states_critic, axis=2)

        self.low_level_buffer.insert(share_obs['low_level'], obs['low_level'], rnn_states_expanded, rnn_states_critic_expanded, actions,
                                     action_log_probs, values, low_rewards_reshaped, masks)
        
    def insert_high_level(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        # rnn_states shape: (n_threads, 1, hidden_size)
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), 1, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), 1, self.hidden_size), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, 1, 1), dtype=np.float32)
        masks[dones == True] = np.float32(0.0)

        # high_level 보상만 추출하여 (n_threads, 1, 1) 모양으로 변환합니다.
        high_rewards_list = [r['high_level'] for r in rewards]
        high_rewards_array = np.array(high_rewards_list) # shape (n_threads,)
        # shape (n_threads, 1, 1) (high_level policy는 n_agents=1이므로)
        high_rewards_reshaped = high_rewards_array.reshape(-1, 1, 1) 

        # rnn_states의 shape (1, 1, 64) -> (1, 1, 1, 64)로 변경
        rnn_states_expanded = np.expand_dims(rnn_states, axis=2)
        rnn_states_critic_expanded = np.expand_dims(rnn_states_critic, axis=2)

        self.high_level_buffer.insert(share_obs['high_level'], obs['high_level'], rnn_states_expanded, rnn_states_critic_expanded, actions,
                                       action_log_probs, values, high_rewards_reshaped, masks)

    def compute_and_train(self):
        # Low Level
        self.low_level_trainer.prep_rollout()

        # Rollout (get_values)을 위해 입력을 Reshape합니다.
        buffer = self.low_level_buffer
        B, N, L, D_rnn = buffer.rnn_states_critic[-1].shape # (1, 4, 1, 64)

        # (B, N, Dim) -> (B*N, Dim)
        share_obs_input = buffer.share_obs[-1].reshape(-1, buffer.share_obs.shape[-1])
        # (B, N, L, D_rnn) -> (B*N, L, D_rnn)
        rnn_states_input = buffer.rnn_states_critic[-1].reshape(-1, L, D_rnn)
        # (B, N, 1) -> (B*N, 1)
        masks_input = buffer.masks[-1].reshape(-1, buffer.masks.shape[-1])

        next_values_low = self.low_level_trainer.policy.get_values(share_obs_input,
                                                                    rnn_states_input,
                                                                    masks_input)
        next_values_low = _t2n(next_values_low)
        self.low_level_buffer.compute_returns(next_values_low, self.low_level_trainer.value_normalizer)

        # High Level
        self.high_level_trainer.prep_rollout()

        # Rollout (get_values)을 위해 입력을 Reshape합니다.
        buffer = self.high_level_buffer
        B, N, L, D_rnn = buffer.rnn_states_critic[-1].shape # (1, 1, 1, 64)

        # (B, N, Dim) -> (B*N, Dim)
        share_obs_input = buffer.share_obs[-1].reshape(-1, buffer.share_obs.shape[-1])
        # (B, N, L, D_rnn) -> (B*N, L, D_rnn)
        rnn_states_input = buffer.rnn_states_critic[-1].reshape(-1, L, D_rnn)
        # (B, N, 1) -> (B*N, 1)
        masks_input = buffer.masks[-1].reshape(-1, buffer.masks.shape[-1])

        next_values_high = self.high_level_trainer.policy.get_values(share_obs_input,
                                                                      rnn_states_input,
                                                                      masks_input)
        next_values_high = _t2n(next_values_high)
        self.high_level_buffer.compute_returns(next_values_high, self.high_level_trainer.value_normalizer)
        
    def save(self):
        """Save high-level and low-level policies."""
        
        # 1. Save High-Level Policy
        policy_actor_high = self.high_level_trainer.policy.actor
        torch.save(policy_actor_high.state_dict(), str(self.save_dir) + "/high_level_actor.pt")
        policy_critic_high = self.high_level_trainer.policy.critic
        torch.save(policy_critic_high.state_dict(), str(self.save_dir) + "/high_level_critic.pt")
        if self.high_level_trainer._use_valuenorm:
            policy_vnorm_high = self.high_level_trainer.value_normalizer
            torch.save(policy_vnorm_high.state_dict(), str(self.save_dir) + "/high_level_vnorm.pt")

        # 2. Save Low-Level Policy
        policy_actor_low = self.low_level_trainer.policy.actor
        torch.save(policy_actor_low.state_dict(), str(self.save_dir) + "/low_level_actor.pt")
        policy_critic_low = self.low_level_trainer.policy.critic
        torch.save(policy_critic_low.state_dict(), str(self.save_dir) + "/low_level_critic.pt")
        if self.low_level_trainer._use_valuenorm:
            policy_vnorm_low = self.low_level_trainer.value_normalizer
            torch.save(policy_vnorm_low.state_dict(), str(self.save_dir) + "/low_level_vnorm.pt")