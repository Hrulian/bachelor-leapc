import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class SquashedGaussian(nn.Module):
    """A squashed Gaussian.
    Samples the output from a Gaussian distribution specified by the input,
    and then squashes the result with a tanh function.
    Finally, the output of the tanh function is scaled and shifted to match the space.

    Can for example be used to enforce certain action bounds of a stochastic policy.

    Attributes:
        scale: The scale of the space-fitting transform.
        loc: The location of the space-fitting transform (for shifting).
    """

    scale: torch.Tensor
    loc: torch.Tensor
    log_std_min: float
    log_std_max: float

    def __init__(
        self,
        space: spaces.Box,
        log_std_min: float = -4,
        log_std_max: float = 2.0,
    ):
        """Initializes the SquashedGaussian module.

        Args:
            space: Space the output should fit to.
            log_std_min: The minimum value for the logarithm of the standard deviation.
            log_std_max: The maximum value for the logarithm of the standard deviation.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Args:
            mean: The mean of the normal distribution.
            log_std: The logarithm of the standard deviation of the normal distribution,
                of the same shape as the mean (i.e., assuming independent dimensions).
                Will be clamped according to the attributes of this class.
            deterministic: If True, the output will just be spacefitting(tanh(mean)),
                no sampling is taking place.

        Returns:
            An output sampled from the SquashedGaussian, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        # Avoid standard deviation exploding or collapsing
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        # Enforce positive standard deviation
        std = torch.exp(log_std)

        if deterministic:
            y = mean
        else:
            # reparameterization trick
            # TODO: Implement the reparameterization trick to sample from a Gaussian
            # Remember that you can create random samples from an arbitrary normal distribution, by
            # scaling and shifting samples from a standard normal distribution.
            # Hint: You can use torch.randn_like to get samples from a standard normal distribution
            y = mean + std * torch.randn_like(mean)

        # Log prob of the normally distributed samples
        log_prob = -0.5 * ((y - mean) / std).pow(2) - log_std - np.log(np.sqrt(2) * np.pi)

        # Squashes the output to be between -1 and 1
        y = torch.tanh(y)

        # Adjust log prob according to tanh squashing and space fitting
        log_prob -= torch.log(self.scale[None, :] * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Rescale the output to be between space.low and space.high
        y_scaled = y * self.scale[None, :] + self.loc[None, :]

        stats = {"gaussian_unsquashed_std": std.prod(dim=-1).mean().item()}

        return y_scaled, log_prob, stats