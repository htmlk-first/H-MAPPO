import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopArt(torch.nn.Module):
    """
    Implements the PopArt normalization layer, as described in:
    "Learning values across many orders of magnitude" (DeepMind, 2016)
    https://arxiv.org/abs/1602.07714
    
    This module is essentially a `nn.Linear` layer that also maintains a
    running estimate of the mean and variance of the targets (e.g., returns)
    it is trying to predict.
    
    It normalizes the targets before computing the loss and automatically
    adjusts its own weights and biases to ensure that the *denormalized*
    output of the network remains consistent as the target statistics change.
    
    This is primarily used as the final layer of the value function (critic).
    """
    
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        """
        Initialize the PopArt layer.
        :param input_shape: (int) Dimension of the input features.
        :param output_shape: (int) Dimension of the output (typically 1 for a value function).
        :param norm_axes: (int) The number of batch dimensions (usually 1).
        :param beta: (float) The momentum coefficient for the running statistics (EMA).
        :param epsilon: (float) A small value added for numerical stability.
        :param device: (torch.device) The device to run on.
        """
        
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)    # Tensor property dictionary

        self.input_shape = input_shape
        self.output_shape = output_shape

        # Standard linear layer parameters (W, b)
        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
        # Running statistics for normalization (mean, variance)
        # These are non-trainable buffers.
        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        
        # Debias term for the exponential moving average
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights and biases of the linear layer."""
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
        # Reset running statistics
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        """
        Performs the forward pass: y = W*x + b.
        This produces the *denormalized* value prediction.
        :param input_vector: (torch.Tensor) Input features from the network.
        :return: (torch.Tensor) The denormalized value prediction.
        """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        # Standard linear transformation
        return F.linear(input_vector, self.weight, self.bias)
    
    @torch.no_grad()
    def update(self, input_vector):
        """
        Updates the running statistics (mean, var) based on a new batch of
        targets and then updates the layer's (W, b) to preserve the output.
        
        This is the core "Pop" (Preserving Outputs) part of the algorithm.
        
        :param input_vector: (torch.Tensor) A batch of new targets (e.g., returns).
        """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        
        # Get the old statistics before the update
        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        # --- Update running statistics (EMA) ---
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        # Update the standard deviation (non-debiased, used for the update rule)
        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)
        
        # Get the new statistics after the update
        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)
        
        # --- Update layer weights and biases ---
        # This operation ensures that if you passed an input `x` through the
        # *old* layer and *denormalized* with the *old* stats, you get the
        # same result as passing `x` through the *new* layer and
        # *denormalizing* with the *new* stats.
        # W_new = W_old * (std_old / std_new)
        # b_new = (std_old * b_old + old_mean - new_mean) / new_stddev
        self.weight = self.weight * old_stddev / new_stddev
        self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

    def debiased_mean_var(self):
        """
        Calculates the debiased (unbiased) mean and variance.
        This corrects for the fact that the EMA starts at zero.
        
        :return: (torch.Tensor, torch.Tensor) The debiased mean and variance.
        """
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        """
        Normalizes a batch of targets (returns) using the running statistics.
        This is used *before* calculating the loss.
        y_norm = (y_raw - mean) / std
        
        :param input_vector: (torch.Tensor) The raw targets (e.g., returns).
        :return: (torch.Tensor) The normalized targets.
        """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        # Normalize: (x - mu) / sigma
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """
        Denormalizes a batch of normalized values (e.g., network predictions)
        back to the original target scale.
        y_raw = y_norm * std + mean
        
        :param input_vector: (torch.Tensor) The normalized values (e.g., network output).
        :return: (np.ndarray) The denormalized values.
        """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        # Denormalize: (x * sigma) + mu
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()

        return out
