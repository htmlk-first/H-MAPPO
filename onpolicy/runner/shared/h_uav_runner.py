import time
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class H_UAVRunner(Runner):
    """
    Runner class specifically designed for Hierarchical MAPPO (H-MAPPO) in the UAV environment.
    It manages two levels of policies: a high-level policy for setting goals and
    a low-level policy for executing actions based on those goals.
    """
    def __init__(self, config):
        """
        Initialize the H_UAVRunner.
        :param config: (dict) Configuration dictionary containing arguments, envs, trainers, buffers, etc.
        """
        
        # --- Initialize essential components from the base Runner ---
        # Instead of calling super().__init__, we directly assign necessary attributes
        # because H-MAPPO requires separate trainers and buffers for high/low levels,
        # which are passed directly in the config, unlike the base Runner setup.
        self.all_args = config['all_args']
        self.envs = config['envs']  # Vectorized environment (e.g., DummyVecEnv)
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']  # Number of low-level agents (UAVs)

        # --- Basic parameters ---
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps # Total training steps
        self.episode_length = self.all_args.episode_length # Max steps per episode
        self.n_rollout_threads = self.all_args.n_rollout_threads # Number of parallel envs
        self.hidden_size = self.all_args.hidden_size # RNN hidden state size
        self.recurrent_N = self.all_args.recurrent_N # Number of RNN layers
        self.use_wandb = self.all_args.use_wandb # Use Weights & Biases for logging
        self.log_interval = self.all_args.log_interval # Frequency for logging training info
        self.save_interval = self.all_args.save_interval # Frequency for saving models

        # --- Directories for logging and saving ---
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)  # TensorboardX writer
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --- H-MAPPO specific components ---
        # The config provides a list of trainers and buffers: [high_level, low_level]
        self.trainer = config['trainer']
        self.trainer = config['trainer']
        self.buffer = config['buffer']
        
        # Assign high and low level trainers and buffers
        self.high_level_trainer = self.trainer[0]
        self.high_level_buffer = self.buffer[0]
        self.low_level_trainer = self.trainer[1]
        self.low_level_buffer = self.buffer[1]

        # Frequency at which the high-level policy acts
        self.high_level_timestep = self.all_args.high_level_timestep

    def run(self):
        """
        Main training loop for H-MAPPO.
        Collects data using both policies, performs training updates.
        """
        self.warmup()   # Initialize buffer with the first observation

        start = time.time()
        # Calculate total number of episodes
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # Apply learning rate decay if enabled
            if self.all_args.use_linear_lr_decay:
                self.high_level_trainer.policy.lr_decay(episode, episodes)
                self.low_level_trainer.policy.lr_decay(episode, episodes)

            # --- Start episode rollout ---
            for step in range(self.episode_length):
                # 1. Sample low-level actions (executed at every step)
                low_values, low_actions, low_action_log_probs, low_rnn_states, low_rnn_states_critic = self.collect_low_level()

                # 2. Sample high-level actions (executed periodically)
                high_actions = None
                if step % self.high_level_timestep == 0:
                    high_values, high_actions, high_action_log_probs, high_rnn_states, high_rnn_states_critic = self.collect_high_level()
                    
                    # Call the environment's method to set goals based on high-level actions
                    # Requires the vec_env (self.envs) to have an `env_method` implementation
                    self.envs.env_method('set_goals', high_actions)
                
                # 3. Step the environment with low-level actions
                # envs.step() returns stacked obs, rewards, dones, infos for all threads
                full_obs, rewards, dones, infos = self.envs.step(low_actions)

                # 4. Unpack observations into high/low levels and obs/share_obs
                obs, share_obs = self.unpack_obs(full_obs)
                
                # 5. Insert low-level data into the low-level buffer
                low_data = obs, share_obs, rewards, dones, infos, low_values, low_actions, low_action_log_probs, low_rnn_states, low_rnn_states_critic
                self.insert_low_level(low_data)

                # 6. Insert high-level data into the high-level buffer (only when high-level acted)
                if step % self.high_level_timestep == 0:
                    high_data = obs, share_obs, rewards, dones, infos, high_values, high_actions, high_action_log_probs, high_rnn_states, high_rnn_states_critic
                    self.insert_high_level(high_data)
            # --- End episode rollout ---
            
            # 7. Compute returns (GAE) and perform training updates for both levels
            low_train_infos, high_train_infos = self.compute_and_train()
            
            # Calculate total steps taken
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # Save models periodically
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
            
            # Log training information periodically
            if (episode % self.log_interval == 0):
                end = time.time()
                print(f"\n Episode {episode}/{episodes}, total num timesteps {total_num_steps} || FPS {int(total_num_steps / (end - start))}")

                # [추가] 캡처한 정보를 log_train으로 전달
                prefixed_low_infos = {f"low_level/{k}": v for k, v in low_train_infos.items()}
                prefixed_high_infos = {f"high_level/{k}": v for k, v in high_train_infos.items()}
                
                self.log_train(prefixed_low_infos, total_num_steps)
                self.log_train(prefixed_high_infos, total_num_steps)

                # [추가] 에피소드 보상 로깅 (uav_runner.py 참고)
                if self.high_level_buffer.rewards.size > 0:
                     avg_high_reward = np.mean(self.high_level_buffer.rewards) * self.episode_length
                     self.log_train({"high_level/episode_reward": avg_high_reward}, total_num_steps)
                
                if self.low_level_buffer.rewards.size > 0:
                     avg_low_reward = np.mean(self.low_level_buffer.rewards) * self.episode_length
                     self.log_train({"low_level/episode_reward": avg_low_reward}, total_num_steps)

    def warmup(self):
        """
        Initializes the buffers with the first observation from the environment.
        """
        # DummyVecEnv.reset() returns a tuple (obs, infos).
        # SubprocVecEnv.reset() returns just obs (due to our fix in the worker).
        full_obs_tuple_or_array = self.envs.reset()
        
        # Handle different return types from VecEnvs
        if isinstance(full_obs_tuple_or_array, tuple):
            # Case: n_rollout_threads = 1 (DummyVecEnv)
            full_obs = full_obs_tuple_or_array[0]
        else:
            # Case: n_rollout_threads > 1 (SubprocVecEnv)
            full_obs = full_obs_tuple_or_array
            
        # Pass only the observation part to unpack_obs.
        obs, share_obs = self.unpack_obs(full_obs)
        
        # Store initial observations in both buffers at step 0.
        self.low_level_buffer.share_obs[0] = share_obs['low_level'].copy()
        self.low_level_buffer.obs[0] = obs['low_level'].copy()

        self.high_level_buffer.share_obs[0] = share_obs['high_level'].copy()
        self.high_level_buffer.obs[0] = obs['high_level'].copy()
        
    def unpack_obs(self, full_obs_stacked):
        """
        Unpacks the observation dictionary received from the environment
        into separate observation (obs) and shared observation (share_obs)
        dictionaries for both high and low levels.
        
        :param full_obs_stacked: (np.ndarray) Array of observation dictionaries
                                 from the vectorized environment. Shape: (n_rollout_threads,).
                                 Each element is {'high_level': ..., 'low_level': ...}.
        :return: (dict, dict) obs_dict, share_obs_dict
        """
        
        # full_obs_stacked shape example: (1,) if n_rollout_threads=1
        
        # 1. High-level obs/share_obs (These are identical and already centralized)
        # d['high_level'] shape: (1, high_obs_dim)
        # high_level_obs shape: (n_rollout_threads, 1, high_obs_dim)
        high_level_obs = np.array([d['high_level'] for d in full_obs_stacked])

        # 2. Low-level obs (Individual agent observations)
        # d['low_level'] shape: (num_agents, low_obs_dim)
        # obs_low_level_stacked shape: (n_rollout_threads, num_agents, low_obs_dim)
        obs_low_level_stacked = np.array([d['low_level'] for d in full_obs_stacked])

        # 3. Low-level share_obs (Centralized observation for the critic)
        # This is constructed by concatenating all low-level observations.
        # Expected dim: share_low_obs_dim = low_obs_dim * num_uavs (as defined in uav_env.py)
        
        n_rollout_threads = len(full_obs_stacked)
        
        # Flatten obs across agents: (n_threads, num_agents, low_obs_dim) -> (n_threads, num_agents * low_obs_dim)
        share_obs_low_level_flat = obs_low_level_stacked.reshape(n_rollout_threads, -1)
        
        # Add a dummy agent dimension: (n_threads, num_agents * low_obs_dim) -> (n_threads, 1, num_agents * low_obs_dim)
        share_obs_low_level_expanded = np.expand_dims(share_obs_low_level_flat, 1)
        
        # Tile across the agent dimension so each agent receives the same centralized state.
        # (n_threads, 1, ...) -> (n_threads, num_agents, ...)
        share_obs_low_level_tiled = np.tile(
            share_obs_low_level_expanded, 
            (1, self.num_agents, 1) # Tile 'num_agents' times along axis 1
        )

        # Return structured dictionaries
        obs = {
            'low_level': obs_low_level_stacked,
            'high_level': high_level_obs
        }
        share_obs = {
            'low_level': share_obs_low_level_tiled,
            'high_level': high_level_obs # High-level share_obs is same as obs
        }
        return obs, share_obs

    @torch.no_grad()
    def collect_low_level(self):
        """
        Collects actions and value predictions from the low-level policy.
        :return: values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        self.low_level_trainer.prep_rollout()   # Set policy to eval mode
        buffer = self.low_level_buffer
        step = buffer.step  # Current step in the buffer
        
        # --- Prepare inputs for the policy ---
        # Reshape inputs from (n_threads, n_agents, dim) to (n_threads * n_agents, dim)
        # as the policy expects a flat batch.
        share_obs_input = buffer.share_obs[step].reshape(-1, buffer.share_obs.shape[-1])
        obs_input = buffer.obs[step].reshape(-1, buffer.obs.shape[-1])
        
        # Reshape RNN states. Shape depends on recurrent_N.
        # Assumes rnn.py expects (B*N, hidden_size) if recurrent_N=1.
        rnn_states_input = buffer.rnn_states[step].reshape(-1, buffer.rnn_states.shape[-1])
        rnn_states_critic_input = buffer.rnn_states_critic[step].reshape(-1, buffer.rnn_states_critic.shape[-1])
        masks_input = buffer.masks[step].reshape(-1, buffer.masks.shape[-1])

        # --- Get actions from policy ---
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.low_level_trainer.policy.get_actions(share_obs_input,
                                                        obs_input,
                                                        rnn_states_input,
                                                        rnn_states_critic_input,
                                                        masks_input)
            
        # --- Reshape outputs back to (n_threads, n_agents, dim) ---
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def collect_high_level(self):
        """
        Collects actions and value predictions from the high-level policy.
        Similar to collect_low_level, but uses the high-level trainer/buffer
        and handles the single high-level agent dimension (num_agents=1).
        :return: values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        self.high_level_trainer.prep_rollout()  # Set policy to eval mode
        buffer = self.high_level_buffer
        step = buffer.step

        # --- Prepare inputs for the policy ---
        # Reshape inputs from (n_threads, 1, dim) -> (n_threads, dim)
        share_obs_input = buffer.share_obs[step].reshape(-1, buffer.share_obs.shape[-1])
        obs_input = buffer.obs[step].reshape(-1, buffer.obs.shape[-1])
        rnn_states_input = buffer.rnn_states[step].reshape(-1, buffer.rnn_states.shape[-1])
        rnn_states_critic_input = buffer.rnn_states_critic[step].reshape(-1, buffer.rnn_states_critic.shape[-1])
        masks_input = buffer.masks[step].reshape(-1, buffer.masks.shape[-1])

        # --- Get actions from policy ---
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.high_level_trainer.policy.get_actions(share_obs_input,
                                                         obs_input,
                                                         rnn_states_input,
                                                         rnn_states_critic_input,
                                                         masks_input)
        
        # --- Reshape outputs back to (n_threads, 1, dim) ---
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert_low_level(self, data):
        """
        Inserts collected low-level experience into the low-level buffer.
        :param data: (tuple) Contains observations, actions, rewards, etc.
        """
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.float32(0.0)

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
        next_values_low = np.array(np.split(next_values_low, self.n_rollout_threads))
        self.low_level_buffer.compute_returns(next_values_low, self.low_level_trainer.value_normalizer)

        # Train network
        self.low_level_trainer.prep_training()
        low_train_infos = self.low_level_trainer.train(self.low_level_buffer)
        self.low_level_buffer.after_update()
        
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
        next_values_high = np.array(np.split(next_values_high, self.n_rollout_threads))
        self.high_level_buffer.compute_returns(next_values_high, self.high_level_trainer.value_normalizer)
        
        # Train network
        self.high_level_trainer.prep_training()
        high_train_infos = self.high_level_trainer.train(self.high_level_buffer)
        self.high_level_buffer.after_update()
        
        # --- 3. Return infos for logging ---
        return low_train_infos, high_train_infos
        
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
            
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                # wandb.log({k: v}, step=total_num_steps) # wandb는 사용 안 함
                pass
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)