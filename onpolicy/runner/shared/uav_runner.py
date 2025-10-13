import time
import numpy as np
import torch
from .base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class UAVRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the UAVs. See parent class for details."""
    def __init__(self, config):
        super(UAVRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "UAV":
                    # buffer.rewards의 모양은 (episode_length, n_threads, n_agents, 1) 입니다.
                    # 우리는 n_threads=1 이므로, 전체 보상의 평균을 계산합니다.
                    avg_reward = np.mean(self.buffer.rewards) * self.episode_length
                    print("average episode reward is {}".format(avg_reward))
                
                self.log_train(train_infos, total_num_steps)

    def warmup(self):
        # reset env
        obs, infos = self.envs.reset()
        
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        share_obs_input = self.buffer.share_obs[step].reshape(-1, self.buffer.share_obs.shape[-1])
        obs_input = self.buffer.obs[step].reshape(-1, self.buffer.obs.shape[-1])
        rnn_states_input = self.buffer.rnn_states[step].reshape(-1, self.buffer.rnn_states.shape[-1])
        rnn_states_critic_input = self.buffer.rnn_states_critic[step].reshape(-1, self.buffer.rnn_states_critic.shape[-1])
        masks_input = self.buffer.masks[step].reshape(-1, self.buffer.masks.shape[-1])

        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(share_obs_input,
                                            obs_input,
                                            rnn_states_input,
                                            rnn_states_critic_input,
                                            masks_input)
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.float32(0.0)

        rewards = np.array(rewards).reshape(self.n_rollout_threads, self.num_agents, 1)

        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        
        # FIX 1: Add the missing dimension for the RNN states before inserting into buffer
        rnn_states = np.expand_dims(rnn_states, axis=2)
        rnn_states_critic = np.expand_dims(rnn_states_critic, axis=2)
        
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks)