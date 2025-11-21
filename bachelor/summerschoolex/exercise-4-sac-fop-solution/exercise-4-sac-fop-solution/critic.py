import torch
import torch.nn as nn
from gymnasium import spaces
from mlp import Mlp

from leap_c.torch.nn.scale import min_max_scaling


class SacCritic(nn.Module):
    """A critic network for Soft Actor-Critic (SAC).
    Consists of multiple Q-networks that estimate the expected return for given state-action pairs.

    Attributes:
        mlp: A list of multi-layer perceptrons (MLPs) that estimate Q-values.
        action_space: The action space of the environment (used for normalizing the actions).
    """

    mlp: nn.ModuleList
    action_space: spaces.Box

    def __init__(
        self,
        action_space: spaces.Box,
        observation_space: spaces.Space,
        num_critics: int,
    ) -> None:
        """Initializes the SAC critic network.

        Args:
            action_space: The action space of the environment (used for normalizing the actions).
            observation_space: The observation space of the environment for the extractors.
            mlp_cfg: The configuration for the MLPs.
            num_critics: The number of critic networks to create.
        """
        super().__init__()

        action_dim = action_space.shape[0]  # type: ignore
        s_dim = observation_space.shape[0]  # type: ignore

        self.mlp = nn.ModuleList(
            [
                Mlp(
                    input_sizes=[s_dim, action_dim],
                    output_sizes=1,
                )
                for _ in range(num_critics)
            ]
        )
        self.action_space = action_space

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        """Returns a list of Q-value estimates for the given state-action pairs."""
        a_norm = min_max_scaling(a, self.action_space)  # type: ignore
        return [mlp(x, a_norm) for mlp in self.mlp]