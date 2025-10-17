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
            if (episode % self.log_interval == 0):
                end = time.time()
                print(f"\n Episode {episode}/{episodes}, total num timesteps {total_num_steps} || FPS {int(total_num_steps / (end - start))}")

    def warmup(self):
        full_obs = self.envs.reset()
        obs, share_obs = self.unpack_obs(full_obs)
        
        self.low_level_buffer.share_obs[0] = share_obs['low_level'].copy()
        self.low_level_buffer.obs[0] = obs['low_level'].copy()

        self.high_level_buffer.share_obs[0] = share_obs['high_level'].copy()
        self.high_level_buffer.obs[0] = obs['high_level'].copy()

    def unpack_obs(self, full_obs_stacked):
        # VecEnv가 반환하는 list of dicts를 dict of lists/arrays로 변환
        obs_list = [d['obs'] for d in full_obs_stacked]
        share_obs_list = [d['share_obs'] for d in full_obs_stacked]

        obs = {
            'low_level': np.array([o['low_level'] for o in obs_list]),
            'high_level': np.array([o['high_level'] for o in obs_list])
        }
        share_obs = {
            'low_level': np.array([s['low_level'] for s in share_obs_list]),
            'high_level': np.array([s['high_level'] for s in share_obs_list])
        }
        return obs, share_obs

    @torch.no_grad()
    def collect_low_level(self):
        self.low_level_trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.low_level_trainer.policy.get_actions(self.low_level_buffer.share_obs[self.low_level_buffer.step],
                                                        self.low_level_buffer.obs[self.low_level_buffer.step],
                                                        self.low_level_buffer.rnn_states[self.low_level_buffer.step],
                                                        self.low_level_buffer.rnn_states_critic[self.low_level_buffer.step],
                                                        self.low_level_buffer.masks[self.low_level_buffer.step])
        return _t2n(value), _t2n(action), _t2n(action_log_prob), _t2n(rnn_states), _t2n(rnn_states_critic)

    @torch.no_grad()
    def collect_high_level(self):
        self.high_level_trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.high_level_trainer.policy.get_actions(self.high_level_buffer.share_obs[self.high_level_buffer.step],
                                                         self.high_level_buffer.obs[self.high_level_buffer.step],
                                                         self.high_level_buffer.rnn_states[self.high_level_buffer.step],
                                                         self.high_level_buffer.rnn_states_critic[self.high_level_buffer.step],
                                                         self.high_level_buffer.masks[self.high_level_buffer.step])
        return _t2n(value), _t2n(action), _t2n(action_log_prob), _t2n(rnn_states), _t2n(rnn_states_critic)

    def insert_low_level(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.low_level_buffer.insert(share_obs['low_level'], obs['low_level'], rnn_states, rnn_states_critic, actions,
                                     action_log_probs, values, rewards['low_level'], masks)

    def insert_high_level(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, 1, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.high_level_buffer.insert(share_obs['high_level'], obs['high_level'], rnn_states, rnn_states_critic, actions,
                                       action_log_probs, values, rewards['high_level'], masks)

    def compute_and_train(self):
        # Low Level
        self.low_level_trainer.prep_rollout()
        next_values_low = self.low_level_trainer.policy.get_values(self.low_level_buffer.share_obs[-1],
                                                                    self.low_level_buffer.rnn_states_critic[-1],
                                                                    self.low_level_buffer.masks[-1])
        next_values_low = _t2n(next_values_low)
        self.low_level_buffer.compute_returns(next_values_low, self.low_level_trainer.value_normalizer)
        
        low_level_train_infos = self.low_level_trainer.train(self.low_level_buffer)
        self.low_level_buffer.after_update()

        # High Level
        self.high_level_trainer.prep_rollout()
        next_values_high = self.high_level_trainer.policy.get_values(self.high_level_buffer.share_obs[-1],
                                                                      self.high_level_buffer.rnn_states_critic[-1],
                                                                      self.high_level_buffer.masks[-1])
        next_values_high = _t2n(next_values_high)
        self.high_level_buffer.compute_returns(next_values_high, self.high_level_trainer.value_normalizer)
        
        high_level_train_infos = self.high_level_trainer.train(self.high_level_buffer)
        self.high_level_buffer.after_update()