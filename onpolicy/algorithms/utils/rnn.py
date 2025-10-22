import torch.nn as nn

class RNNLayer(nn.Module):
    """
    A PyTorch module for a recurrent layer (specifically GRU)
    with custom weight initialization and optional LayerNorm.
    This module is designed to handle both single-step (rollout)
    and sequence (training) inputs.
    """
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        """
        Initialize the RNNLayer.
        :param inputs_dim: (int) Dimension of the input features.
        :param outputs_dim: (int) Dimension of the hidden state and output features.
        :param recurrent_N: (int) Number of recurrent layers (e.g., stacked GRU layers).
        :param use_orthogonal: (bool) Whether to use orthogonal initialization for weights.
        """
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        # Create the GRU layer
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        
        # Initialize weights and biases
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                # Initialize biases to 0
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                # Initialize weights
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
        # Layer normalization for the output
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        Forward pass for the RNN.
        This function handles two modes based on input shapes:
        1.  **Training (is_training=True):** - x: (L*N, input_size) - A sequence of length L for N agents.
            - hxs: (N, n_layers, hidden_size) - The initial hidden state for N agents.
            - masks: (L*N, 1) - Masks for each step in the sequence.
        2.  **Rollout (is_training=False):**
            - x: (B*N, input_size) - A single step for B*N agents (B=1 usually).
            - hxs: (B*N, n_layers, hidden_size) - The current hidden state.
            - masks: (B*N, 1) - Mask for the current step.
        
        :param x: (torch.Tensor) Input features.
        :param hxs: (torch.Tensor) Hidden state.
        :param masks: (torch.Tensor) Done masks (0.0 for terminal states).
        :return x: (torch.Tensor) Output features from the RNN.
        :return hxs: (torch.Tensor) The new hidden state.
        """
        
        # This squeeze is specific to this codebase.
        # If the hidden state from the buffer has an extra dimension 
        # (e.g., [N, 1, HiddenDim] when recurrent_N=1),
        # this removes it, making hxs [N, HiddenDim].
        if hxs.dim() == 3: # (num_agents, num_layers, hidden_dim) -> (num_agents, hidden_dim)
            hxs = hxs.squeeze(1)

        # Detect mode: In training, x (L*N) and hxs (N) have different batch sizes.
        # In rollout, x (B*N) and hxs (B*N) have the same batch size.
        is_training = x.size(0) != hxs.size(0)

        if is_training:
            # --- Training Mode (Sequence Input) ---
            # L = sequence length, N = batch size (num_agents)
            T, N = x.size(0) // hxs.size(0), hxs.size(0)
            
            # Reshape x to (L, N, input_size) - PyTorch RNNs expect (SeqLen, Batch, Dim)
            x = x.view(T, N, x.size(1))
            
            # Reshape hxs to (num_layers, N, hidden_size)
            # Assumes hxs was [N, HiddenDim], adds the layer dim.
            hxs = hxs.unsqueeze(0) # Add layer dimension: (1, N, hidden_size)
            
            # Reshape masks to (L, N, 1)
            masks = masks.view(T, N, 1)
            # Apply mask to each step of the sequence
            # This prevents gradients from flowing across episode boundaries.
            x = x * masks
            
        else:
            # --- Rollout Mode (Single Step Input) ---
            # Reshape x to (1, B*N, input_size) - (SeqLen=1, Batch, Dim)
            x = x.unsqueeze(0)
            # Reshape hxs to (num_layers, B*N, hidden_size)
            hxs = hxs.unsqueeze(0) # hxs (4,64) -> (1,4,64)
            
            # Reshape masks for broadcasting
            masks = masks.transpose(0, 1) # (N,1) -> (1,N)
            
            # Apply masks to input and hidden state
            # If mask is 0 (episode done), this resets the hidden state (hxs)
            # to zero for the *next* step and zeros the current input (x).
            x = x * masks.unsqueeze(-1) # (1,N,1) for broadcasting with x
            hxs = hxs * masks.unsqueeze(-1) # (1,N,1) for broadcasting with hxs

        # Run the GRU
        # x shape: (L, N, dim) or (1, N, dim)
        # hxs shape: (num_layers, N, dim)
        x, hxs = self.rnn(x, hxs)
        
        # --- Post-processing ---
        if is_training:
            # Reshape x back to a flat batch (L*N, hidden_size)
            x = x.view(T * N, -1)
        else:
            # Squeeze the sequence dimension (L=1)
            x = x.squeeze(0)
        
        # Squeeze the layer dimension from the output hidden state
        # (num_layers, N, hidden_size) -> (N, hidden_size)
        hxs = hxs.squeeze(0)

        # Apply layer normalization to the output features
        x = self.norm(x)

        return x, hxs