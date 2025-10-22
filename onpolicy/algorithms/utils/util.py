import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    """
    Initialize the weights and biases of a neural network module.
    This function applies a specified weight initialization method and
    a bias initialization method to the module's parameters.
    
    :param module: (nn.Module) The PyTorch module to initialize (e.g., nn.Linear).
    :param weight_init: (function) The function to use for weight initialization
                       (e.g., nn.init.orthogonal_).
    :param bias_init: (function) The function to use for bias initialization
                     (e.g., lambda x: nn.init.constant_(x, 0)).
    :param gain: (float or int) The gain parameter, often used by initialization
                 methods like orthogonal_ or xavier_uniform_.
    :return: (nn.Module) The initialized module.
    """
    # Apply the weight initialization function with gain
    weight_init(module.weight.data, gain=gain)
    # Apply the bias initialization function
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    """
    Create N deep copies of a given PyTorch module.
    This is used, for example, in MLPBase to create multiple hidden layers.
    
    :param module: (nn.Module) The module to be copied.
    :param N: (int) The number of copies to create.
    :return: (nn.ModuleList) A list containing N deep copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    """
    Ensure the input is a PyTorch tensor.
    If the input is a NumPy array, it converts it to a PyTorch tensor.
    If it's already a tensor, it returns it as is.
    
    :param input: (np.ndarray or torch.Tensor) The input data.
    :return: (torch.Tensor) The data as a PyTorch tensor.
    """
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
