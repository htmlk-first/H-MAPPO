import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    """
    A multi-layer perceptron (MLP) module consisting of several fully connected
    layers with activation functions and layer normalization.
    """
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        """
        Initialize the MLPLayer.
        :param input_dim: (int) Dimension of the input features.
        :param hidden_size: (int) Dimension of the hidden layers.
        :param layer_N: (int) Number of hidden layers to create.
        :param use_orthogonal: (bool) Whether to use orthogonal initialization.
        :param use_ReLU: (bool) Whether to use ReLU (True) or Tanh (False) activation.
        """
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        # Determine activation function and initialization method
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        # Helper function to initialize a layer
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # Define the first layer (input layer)
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        
        # Define a template for hidden layers
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        
        # Create N clones of the hidden layer
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        """
        Forward pass through the MLP.
        :param x: (torch.Tensor) Input features.
        :return: (torch.Tensor) Output features.
        """
        # Pass through the first layer
        x = self.fc1(x)
        
        # Pass through all N hidden layers
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    """
    The base class for the MLP observation encoder.
    This module wraps the MLPLayer and adds optional input feature normalization.
    It is used by the actor and critic networks when observations are flat vectors.
    """
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        """
        Initialize the MLPBase.
        :param args: (argparse.Namespace) Arguments containing model hyperparameters.
        :param obs_shape: (tuple) The shape of the observation (assumed to be [dim,]).
        :param cat_self: (bool) Not used in this implementation but kept for compatibility.
        :param attn_internal: (bool) Not used in this implementation.
        """
        super(MLPBase, self).__init__()

        # Store hyperparameters
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # Get the input dimension from the observation shape
        obs_dim = obs_shape[0]

        # Optional LayerNorm for the input features
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # Create the core MLP network
        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

        # [Modification for H-MAPPO]
        # Add an 'output_dim' attribute. This is essential for the H_Actor 
        # (in h_actor_critic.py) to know the dimension of the features produced
        # by this encoder, allowing it to correctly concatenate features from multiple encoders.
        self.output_dim = self.hidden_size

    def forward(self, x):
        """
        Forward pass for the MLPBase.
        :param x: (torch.Tensor) Input observation vector.
        :return: (torch.Tensor) Output feature vector.
        """
        # Apply feature normalization if enabled
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        # Pass input through the MLP
        x = self.mlp(x)

        return x