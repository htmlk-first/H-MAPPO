import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space

class H_Actor(nn.Module):
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
            # High-level policy
            self.base = MLPBase(args, obs_shape)
            gru_input_dim = self.base.output_dim
        else: # Low-level policy
            self_obs_dim = 4
            other_uav_dim = 2 * (self.num_agents - 1)
            goal_dim = 2
            
            self.self_encoder = MLPBase(args, (self_obs_dim,))
            self.other_uav_encoder = MLPBase(args, (other_uav_dim,))
            self.goal_encoder = MLPBase(args, (goal_dim,))
            
            gru_input_dim = self.self_encoder.output_dim + self.other_uav_encoder.output_dim + self.goal_encoder.output_dim
            self.base = None
        
        if self._use_recurrent_policy:
            self.rnn = RNNLayer(gru_input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, args.gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if self.base is not None:
            # High-level
            actor_features = self.base(obs)
        else:
            # Low-level
            other_uav_obs_dim = 2 * (self.num_agents - 1)
            self_obs = obs[:, :4]
            other_uav_obs = obs[:, 4 : 4 + other_uav_obs_dim]
            goal_obs = obs[:, 4 + other_uav_obs_dim:]

            self_features = self.self_encoder(self_obs)
            other_uav_features = self.other_uav_encoder(other_uav_obs)
            goal_features = self.goal_encoder(goal_obs)
            
            actor_features = torch.cat([self_features, other_uav_features, goal_features], dim=-1)

        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

class H_Critic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super(H_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        
        # use_attn 인자가 없으므로 MLPBase 호출을 그대로 사용
        self.base = MLPBase(args, share_obs_shape)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, share_obs, rnn_states, masks):
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        critic_features = self.base(share_obs)
        
        if self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_states