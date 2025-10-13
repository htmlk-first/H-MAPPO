import torch.nn as nn

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        Compute hidden states from inputs.
        :param x: (B*N, input_size) for rollout, (L*N, input_size) for training.
        :param hxs: (B*N, n_layers, hidden_size) for rollout, (N, n_layers, hidden_size) for training.
        :param masks: (B*N, 1) for rollout, (L*N, 1) for training.
        :return hxs: (B*N or L*N, hidden_size)
        """
        # Squeeze the layer dimension if it exists, making hxs 2D
        if hxs.dim() == 3: # (num_agents, num_layers, hidden_dim) -> (num_agents, hidden_dim)
            hxs = hxs.squeeze(1)

        is_training = x.size(0) != hxs.size(0)

        if is_training:
            # Training logic: x is a sequence, hxs is the initial state for the sequence batch
            T, N = x.size(0) // hxs.size(0), hxs.size(0)
            x = x.view(T, N, x.size(1))
            hxs = hxs.unsqueeze(0) # Add layer dimension: (1, N, hidden_size)
            
            # Reshape masks to match the sequence structure of x
            masks = masks.view(T, N, 1)
            # Apply mask to each step of the sequence
            x = x * masks
            # No need to mask hxs directly if using packed sequence, but easier to just mask x
        else:
            # Rollout logic: x is a single step for each agent
            x = x.unsqueeze(0)
            hxs = hxs.unsqueeze(0) # hxs (4,64) -> (1,4,64) which is correct for GRU
            
            # masks shape is (N, 1), we need to apply it to x which is (1, N, feature_dim)
            # and hxs which is (1, N, hidden_size). A simple multiplication will broadcast correctly.
            masks = masks.transpose(0, 1) # (N,1) -> (1,N)
            x = x * masks.unsqueeze(-1) # (1,N,1) for broadcasting with x
            hxs = hxs * masks.unsqueeze(-1) # (1,N,1) for broadcasting with hxs


        x, hxs = self.rnn(x, hxs)
        
        if is_training:
            # Reshape x back to a flat batch for the actor/critic heads
            x = x.view(T * N, -1)
        else:
            x = x.squeeze(0)
        
        hxs = hxs.squeeze(0)

        # Apply layer normalization
        x = self.norm(x)

        return x, hxs