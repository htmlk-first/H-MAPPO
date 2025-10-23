import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space

class H_Actor(nn.Module):
    """
    Hierarchical Actor network.
    This class is designed to handle both high-level and low-level policies.
    - For high-level policy (num_agents=1), it uses a standard MLPBase.
    - For low-level policy (num_agents > 1), it uses a custom MLP structure
      that encodes self, other UAVs, and goal information separately.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(H_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents
        obs_shape = get_shape_from_obs_space(obs_space)
        
        if self.num_agents == 1:
            # High-level policy: Use a standard MLP base
            self.base = MLPBase(args, obs_shape)
            gru_input_dim = self.base.output_dim
        else:
            # Low-level policy: Custom encoder structure
            # Define dimensions for different parts of the low-level observation
            # (목표 설계 반영) S_low 변경 (p_los 추가)
            self_obs_dim = 5    # [pos_x, pos_y, energy, sinr, p_los]
            
            other_uav_dim = 2 * (self.num_agents - 1)   # [rel_pos_x, rel_pos_y] for each other UAV
            goal_dim = 2    # [rel_goal_x, rel_goal_y]
            
            # Create separate encoders for each observation component
            self.self_encoder = MLPBase(args, (self_obs_dim,))
            self.other_uav_encoder = MLPBase(args, (other_uav_dim,))
            self.goal_encoder = MLPBase(args, (goal_dim,))
            
            # The input to the RNN will be the concatenated features from all encoders
            gru_input_dim = self.self_encoder.output_dim + self.other_uav_encoder.output_dim + self.goal_encoder.output_dim
            self.base = None    # Set base to None to indicate custom structure
        
        # Action output layer
        if self._use_recurrent_policy:
            self.rnn = RNNLayer(gru_input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, args.gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        # Ensure all inputs are torch tensors on the correct device
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if self.base is not None:
            # High-level policy forward pass
            actor_features = self.base(obs)
        else:
            # Low-level policy forward pass
            # Split the observation tensor into its components
            other_uav_obs_dim = 2 * (self.num_agents - 1)
            
            # (목표 설계 반영) S_low 변경 (p_los 추가)에 따른 인덱싱 수정
            self_obs = obs[:, :5]
            other_uav_obs = obs[:, 5 : 5 + other_uav_obs_dim]
            goal_obs = obs[:, 5 + other_uav_obs_dim:]

            # Pass components through their respective encoders
            self_features = self.self_encoder(self_obs)
            other_uav_features = self.other_uav_encoder(other_uav_obs)
            goal_features = self.goal_encoder(goal_obs)
            
            # Concatenate features to form the input for the RNN
            actor_features = torch.cat([self_features, other_uav_features, goal_features], dim=-1)

        if self._use_recurrent_policy:
            # Pass features through RNN
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        (This method is required by the PPO update logic).
        """
        # Ensure all inputs are torch tensors on the correct device
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv) # Add action tensor check
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        if self.base is not None:
            # High-level policy forward pass
            actor_features = self.base(obs)
        else:
            # Low-level policy forward pass
            # Split the observation tensor into its components
            other_uav_obs_dim = 2 * (self.num_agents - 1)
            
            # (목표 설계 반영) S_low 변경 (p_los 추가)에 따른 인덱싱 수정
            self_obs = obs[:, :5]
            other_uav_obs = obs[:, 5 : 5 + other_uav_obs_dim]
            goal_obs = obs[:, 5 + other_uav_obs_dim:]

            # Pass components through their respective encoders
            self_features = self.self_encoder(self_obs)
            other_uav_features = self.other_uav_encoder(other_uav_obs)
            goal_features = self.goal_encoder(goal_obs)
            
            # Concatenate features to form the input for the RNN
            actor_features = torch.cat([self_features, other_uav_features, goal_features], dim=-1)

        if self._use_recurrent_policy:
            # Pass features through RNN
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # [KEY CHANGE] Call evaluate_actions on the ACTLayer
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action,
                                                                   available_actions,
                                                                   active_masks=active_masks)
                                                                   
        return action_log_probs, dist_entropy

class H_Critic(nn.Module):
    """
    Hierarchical Critic network.
    This critic uses a standard MLPBase for both high-level and low-level
    policies, as it always receives a centralized 'share_obs'.
    """
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super(H_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        
        # The base is always an MLP as the share_obs is a flat vector
        self.base = MLPBase(args, share_obs_shape)

        if self._use_recurrent_policy:
            # RNN layer
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # Initialization function for the output layer
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        # Value output layer
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, share_obs, rnn_states, masks):
        # Ensure all inputs are torch tensors on the correct device
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Pass centralized observation through the base network
        critic_features = self.base(share_obs)
        
        if self._use_recurrent_policy:
            # Pass features through RNN
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            
        # Get state value prediction
        values = self.v_out(critic_features)
        return values, rnn_states