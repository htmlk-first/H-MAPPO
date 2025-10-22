import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase.
This file defines:
1.  `Fixed...` classes (FixedCategorical, FixedNormal, FixedBernoulli):
    These inherit from standard PyTorch distributions but modify methods like `sample()` 
    and `log_probs()` to ensure output tensors have a consistent shape 
    (e.g., adding an extra dimension), which simplifies downstream processing 
    in the policy update logic.
2.  `nn.Module` wrappers (Categorical, DiagGaussian, Bernoulli):
    These are PyTorch modules (like `nn.Linear`) that act as the final "head" 
    of a network. They take features as input, apply a linear layer, 
    and output one of the `Fixed...` distribution objects.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    """
    A wrapper for `torch.distributions.Categorical` that ensures:
    1.  `sample()` returns a tensor with a trailing dimension of 1.
    2.  `log_probs()` sums log probabilities and returns a tensor with a 
        trailing dimension of 1.
    """
    def sample(self):
        # Call parent sample() and add a new dimension at the end.
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        """
        :param actions: (torch.Tensor) Actions, typically shape [BatchSize, 1].
        :return: (torch.Tensor) Log probabilities, shape [BatchSize, 1].
        """
        # Squeeze the last dimension of actions for the parent's log_prob.
        # Then, view/sum/unsqueeze to ensure the output is [BatchSize, 1].
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        # Return the action with the highest probability (argmax), keeping the
        # trailing dimension.
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    """
    A wrapper for `torch.distributions.Normal` (specifically for diagonal covariance).
    Ensures `log_probs` and `entropy` sum across the action dimensions and
    return a tensor with a trailing dimension of 1.
    """
    def log_probs(self, actions):
        # Sum log probabilities across the action dimensions (dim -1).
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        # Sum entropy across the action dimensions.
        return super().entropy().sum(-1)

    def mode(self):
        # The mode of a Normal distribution is its mean.
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    """
    A wrapper for `torch.distributions.Bernoulli`.
    Ensures `log_probs` and `entropy` sum across dimensions and
    return a tensor with a trailing dimension of 1.
    """
    def log_probs(self, actions):
        # Sum log probabilities across the action dimensions.
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        # Sum entropy across the action dimensions.
        return super().entropy().sum(-1)

    def mode(self):
        # The mode is 1 if probs > 0.5, else 0.
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    """
    A `nn.Module` wrapper that creates a `FixedCategorical` distribution.
    This module applies a linear layer to input features to produce logits.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        """
        :param num_inputs: (int) Dimension of input features.
        :param num_outputs: (int) Number of discrete actions (dimension of logits).
        :param use_orthogonal: (bool) Whether to use orthogonal initialization.
        :param gain: (float) Gain for orthogonal initialization.
        """
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            # Initialize the linear layer
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        """
        :param x: (torch.Tensor) Input features [BatchSize, num_inputs].
        :param available_actions: (torch.Tensor, optional) Mask for available actions.
                                  Shape [BatchSize, num_outputs].
        :return: (FixedCategorical) The distribution object.
        """
        # Compute logits
        x = self.linear(x)
        if available_actions is not None:
            # Mask unavailable actions by setting their logits to a very small number
            x[available_actions == 0] = -1e10
        # Create the categorical distribution from logits
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """
    A `nn.Module` wrapper that creates a `FixedNormal` distribution 
    with a diagonal covariance matrix (i.e., independent actions).
    This module computes the mean of the actions via a linear layer
    and uses a separate trainable parameter for the log standard deviation.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        """
        :param num_inputs: (int) Dimension of input features.
        :param num_outputs: (int) Dimension of the continuous action space.
        :param use_orthogonal: (bool) Whether to use orthogonal initialization.
        :param gain: (float) Gain for orthogonal initialization.
        """
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            # Initialize the linear layer for the mean
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        # Linear layer to compute the action means
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        
        # Trainable parameter for the log standard deviation (log_std).
        # This is shared across the batch.
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        """
        :param x: (torch.Tensor) Input features [BatchSize, num_inputs].
        :return: (FixedNormal) The distribution object.
        """
        # Compute action mean
        action_mean = self.fc_mean(x)

        # Compute action log standard deviation
        # An ugly hack for KFAC implementation: passing zeros to AddBias
        # to get the trainable logstd parameter.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        
        # Create the Normal distribution
        # .exp() converts log_std to std
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    """
    A `nn.Module` wrapper that creates a `FixedBernoulli` distribution.
    This module applies a linear layer to input features to produce logits.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            # Initialize the linear layer
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        """
        :param x: (torch.Tensor) Input features [BatchSize, num_inputs].
        :return: (FixedBernoulli) The distribution object.
        """
        # Compute logits
        x = self.linear(x)
        
        # Create the Bernoulli distribution from logits
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    """
    A helper module to add a trainable bias vector (self._bias) to an input tensor 'x'.
    Used by `DiagGaussian` to hold the trainable log_std parameter.
    """
    def __init__(self, bias):
        super(AddBias, self).__init__()
        # self._bias is the trainable parameter
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        # Add the bias to the input
        return x + bias
