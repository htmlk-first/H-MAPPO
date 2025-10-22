import torch
import torch.nn as nn
from .distributions import Categorical, DiagGaussian
from .util import init

class ACTLayer(nn.Module):
    """
    Action Layer (ACTLayer) for the policy network.
    This module takes the final features from the base network (MLP/RNN)
    and outputs an action distribution appropriate for the environment's action space.
    It handles various action space types: Discrete, MultiDiscrete, Box (Continuous),
    and a custom hybrid 'Tuple' space (Mixed).
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        """
        Initialize the ACTLayer.
        :param action_space: (gym.Space) The action space of the environment.
        :param inputs_dim: (int) The dimension of the input features.
        :param use_orthogonal: (bool) Whether to use orthogonal initialization.
        :param gain: (float) The gain for the final layer initialization.
        """
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.continuous_action = False

        # Check action space type by class name for compatibility.
        action_space_class_name = action_space.__class__.__name__

        if action_space_class_name == "Tuple":
            # This branch handles hybrid (mixed) action spaces, defined as a Tuple.
            # The comment below suggests it's for a (Box, MultiDiscrete) space.
            self.mixed_action = True
            cont_action_space, disc_action_space = action_space[0], action_space[1]
            
            # --- Continuous action setup ---
            self.continuous_dim = cont_action_space.shape[0]
            # Create a Gaussian distribution head for the continuous part.
            self.action_outs_cont = DiagGaussian(inputs_dim, self.continuous_dim, use_orthogonal, gain)

            # --- Discrete action setup ---
            self.discrete_dims = disc_action_space.nvec # Get dimensions for each discrete action
            # Create a list of Categorical distribution heads for the discrete part.
            self.action_outs_disc = nn.ModuleList([Categorical(inputs_dim, dim, use_orthogonal, gain) for dim in self.discrete_dims])

        elif action_space_class_name == "Box":
            # Standard continuous action space (e.g., velocities, positions).
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)

        elif action_space_class_name == "MultiDiscrete":
            # Standard multi-discrete action space (e.g., multiple categorical choices).
            # This is used by the provided uav_env.py for both high and low levels.
            self.multi_discrete = True
            self.action_dims = action_space.nvec    # Vector of dimensions for each discrete action
            # Create a list of Categorical heads, one for each dimension.
            self.action_outs = nn.ModuleList([Categorical(inputs_dim, action_dim, use_orthogonal, gain) for action_dim in self.action_dims])

        else:  # Discrete
            # Standard single discrete action space.
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Forward pass: Generate actions from input features.
        Used during rollouts (action sampling).
        :param x: (torch.Tensor) Input features from the base network.
        :param available_actions: (torch.Tensor) Optional mask for available discrete actions.
        :param deterministic: (bool) Whether to sample (False) or take the mode (True) of the distribution.
        :return actions: (torch.Tensor) The sampled or deterministic actions.
        :return action_log_probs: (torch.Tensor) The log probabilities of the taken actions.
        """
        if self.mixed_action:
            # --- Continuous Action ---
            cont_dist = self.action_outs_cont(x)
            cont_action = cont_dist.mode() if deterministic else cont_dist.sample()
            cont_action_log_probs = cont_dist.log_probs(cont_action)

            # --- Discrete Actions ---
            disc_actions = []
            disc_action_log_probs = []
            for action_out in self.action_outs_disc:
                disc_dist = action_out(x, available_actions)
                disc_action = disc_dist.mode() if deterministic else disc_dist.sample()
                disc_action_log_probs.append(disc_dist.log_probs(disc_action))
                disc_actions.append(disc_action)
            
            # --- Combine actions and log_probs into single tensors ---
            # Concatenate continuous and discrete actions
            action = torch.cat([cont_action] + [da.float() for da in disc_actions], dim=-1)
            # Sum log probabilities (log(P(a,b)) = log(P(a)) + log(P(b)))
            action_log_probs = cont_action_log_probs + torch.cat(disc_action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            return action, action_log_probs
        
        elif self.continuous_action:
            action_dist = self.action_out(x)
            action = action_dist.mode() if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_probs(action)
            return action, action_log_probs
        
        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            # Iterate over each discrete action head
            for action_out in self.action_outs:
                action_dist = action_out(x, available_actions)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            # Concatenate all actions into one tensor
            action = torch.cat(actions, dim=-1)
            # Sum all log probabilities
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            return action, action_log_probs
        
        else:  # Discrete
            action_dist = self.action_out(x, available_actions)
            action = action_dist.mode() if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_probs(action)
            return action, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Evaluate actions: Compute log probabilities and entropy for given actions.
        Used during training (PPO update).
        :param x: (torch.Tensor) Input features from the base network.
        :param action: (torch.Tensor) The actions taken (from the replay buffer).
        :param available_actions: (torch.Tensor) Optional mask for available discrete actions.
        :param active_masks: (torch.Tensor) Optional mask for active agents.
        :return action_log_probs: (torch.Tensor) The log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) The entropy of the action distribution.
        """
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
                # Get the specific discrete action from the buffer
                action_slice = disc_actions[:, i].long().unsqueeze(-1)
                log_prob = disc_dist.log_probs(action_slice)
                disc_action_log_probs_list.append(log_prob)
                dist_entropy += disc_dist.entropy().mean()

            # --- Combine ---
            action_log_probs = cont_action_log_probs + torch.cat(disc_action_log_probs_list, dim=-1).sum(dim=-1, keepdim=True)
            return action_log_probs, dist_entropy

        elif self.continuous_action:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            dist_entropy = action_dist.entropy().mean()
            return action_log_probs, dist_entropy
        
        elif self.multi_discrete:
            action_log_probs = []
            dist_entropy = 0
            # Iterate over each discrete action head
            for i, action_out in enumerate(self.action_outs):
                action_dist = action_out(x, available_actions)
                # Get the specific discrete action from the buffer for this dimension
                action_log_probs.append(action_dist.log_probs(action[:, i].unsqueeze(-1)))
                dist_entropy += action_dist.entropy().mean()
            # Sum log probabilities across all dimensions
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            return action_log_probs, dist_entropy
        
        else:  # Discrete
            action_dist = self.action_out(x, available_actions)
            action_log_probs = action_dist.log_probs(action)
            # Calculate entropy, considering active masks if provided
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy().mean()
            return action_log_probs, dist_entropy