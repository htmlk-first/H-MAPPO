import torch.nn as nn
from .util import init

"""CNN Modules and utils."""

class Flatten(nn.Module):
    """
    A simple PyTorch module to flatten a tensor.
    It reshapes the input tensor 'x' from [BatchSize, C, H, W] (or other)
    to [BatchSize, -1], where -1 infers the product of all other dimensions.
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    """
    A single CNN layer followed by fully connected layers.
    This module processes image-like observations.
    """
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        """
        Initialize the CNNLayer.
        :param obs_shape: (tuple) The shape of the observation (Channels, Height, Width).
        :param hidden_size: (int) The size of the output feature vector.
        :param use_orthogonal: (bool) Whether to use orthogonal initialization.
        :param use_ReLU: (bool) Whether to use ReLU (True) or Tanh (False) activation.
        :param kernel_size: (int) The kernel size for the convolutional layer.
        :param stride: (int) The stride for the convolutional layer.
        """
        super(CNNLayer, self).__init__()

        # Determine activation function and initialization method
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        # Helper function to initialize layers
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        # Define the network architecture
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        """
        Forward pass for the CNNLayer.
        :param x: (torch.Tensor) Input image tensor (BatchSize, C, H, W).
        :return: (torch.Tensor) Output feature vector (BatchSize, hidden_size).
        """
        # Normalize pixel values from [0, 255] to [0, 1.0]
        x = x / 255.0
        # Pass through the network
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    """
    The base class for the CNN observation encoder.
    This is a wrapper around CNNLayer, intended to be used as the 'base'
    network in the actor-critic architecture.
    """
    def __init__(self, args, obs_shape):
        """
        Initialize the CNNBase.
        :param args: (argparse.Namespace) Arguments containing model hyperparameters.
        :param obs_shape: (tuple) The shape of the observation (C, H, W).
        """
        super(CNNBase, self).__init__()

        # Store hyperparameters
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        # Create the CNNLayer instance
        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        """
        Forward pass for the CNNBase.
        :param x: (torch.Tensor) Input image tensor.
        :return: (torch.Tensor) Output feature vector.
        """
        x = self.cnn(x)
        return x
