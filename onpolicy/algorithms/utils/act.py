# onpolicy/algorithms/utils/act.py

import torch
import torch.nn as nn
from .distributions import Categorical, DiagGaussian
from .util import init

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions and action log probabilities.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.continuous_action = False

        if isinstance(action_space, tuple):
            # Our Custom UAV Hybrid Action Space: (Box, MultiDiscrete)
            self.mixed_action = True
            cont_action_space, disc_action_space = action_space[0], action_space[1]
            
            # Continuous action setup
            self.continuous_dim = cont_action_space.shape[0]
            self.action_outs_cont = DiagGaussian(inputs_dim, self.continuous_dim, use_orthogonal, gain)

            # Discrete action setup
            self.discrete_dims = disc_action_space.nvec
            self.action_outs_disc = nn.ModuleList([Categorical(inputs_dim, dim, use_orthogonal, gain) for dim in self.discrete_dims])

        else:
            from gymnasium.spaces import Box, Discrete, MultiDiscrete
            if isinstance(action_space, Box):
                self.continuous_action = True
                action_dim = action_space.shape[0]
                self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
            elif isinstance(action_space, MultiDiscrete):
                self.multi_discrete = True
                self.action_dims = action_space.nvec
                self.action_outs = nn.ModuleList([Categorical(inputs_dim, action_dim, use_orthogonal, gain) for action_dim in self.action_dims])
            else:  # Discrete
                action_dim = action_space.n
                self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False):
        if self.mixed_action:
            # --- Continuous Action ---
            cont_dist = self.action_outs_cont(x)
            if deterministic:
                cont_action = cont_dist.mode()
            else:
                cont_action = cont_dist.sample()
            cont_action_log_probs = cont_dist.log_probs(cont_action)

            # --- Discrete Actions ---
            disc_actions = []
            disc_action_log_probs = []
            for action_out in self.action_outs_disc:
                disc_dist = action_out(x, available_actions)
                if deterministic:
                    disc_action = disc_dist.mode()
                else:
                    disc_action = disc_dist.sample()
                
                disc_action_log_probs.append(disc_dist.log_probs(disc_action))
                disc_actions.append(disc_action)
            
            # --- Combine actions and log_probs into single tensors ---
            action = torch.cat([cont_action] + [da.float() for da in disc_actions], dim=-1)
            action_log_probs = cont_action_log_probs + torch.cat(disc_action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            
            return action, action_log_probs
        
        # Original logic for non-hybrid spaces
        elif self.continuous_action:
            action_dist = self.action_out(x)
            if deterministic: action = action_dist.mode()
            else: action = action_dist.sample()
            action_log_probs = action_dist.log_probs(action)
            return action, action_log_probs
        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x, available_actions)
                if deterministic: action = action_dist.mode()
                else: action = action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            action = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            return action, action_log_probs
        else:  # Discrete
            action_dist = self.action_out(x, available_actions)
            if deterministic: action = action_dist.mode()
            else: action = action_dist.sample()
            action_log_probs = action_dist.log_probs(action)
            return action, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        if self.mixed_action:
            # --- Split the flattened action tensor ---
            cont_action = action[:, :self.continuous_dim]
            disc_actions = action[:, self.continuous_dim:]

            # --- Evaluate Continuous Action ---
            cont_dist = self.action_outs_cont(x)
            cont_action_log_probs = cont_dist.log_probs(cont_action)
            dist_entropy = cont_dist.entropy().mean()

            # --- Evaluate Discrete Actions ---
            disc_action_log_probs_list = []
            for i, action_out in enumerate(self.action_outs_disc):
                disc_dist = action_out(x, available_actions)
                action_slice = disc_actions[:, i].long().unsqueeze(-1)
                log_prob = disc_dist.log_probs(action_slice)
                disc_action_log_probs_list.append(log_prob)
                dist_entropy += disc_dist.entropy().mean()

            # --- Combine ---
            action_log_probs = cont_action_log_probs + torch.cat(disc_action_log_probs_list, dim=-1).sum(dim=-1, keepdim=True)
            
            return action_log_probs, dist_entropy

        # Original logic for non-hybrid spaces
        elif self.continuous_action:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            dist_entropy = action_dist.entropy().mean()
            return action_log_probs, dist_entropy
        elif self.multi_discrete:
            action_log_probs = []
            dist_entropy = 0
            for i, action_out in enumerate(self.action_outs):
                action_dist = action_out(x, available_actions)
                action_log_probs.append(action_dist.log_probs(action[:, i].unsqueeze(-1)))
                dist_entropy += action_dist.entropy().mean()
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            return action_log_probs, dist_entropy
        else:  # Discrete
            action_dist = self.action_out(x, available_actions)
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy().mean()
            return action_log_probs, dist_entropy