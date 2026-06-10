"""This file contains utility functions that are used in the training loop."""

import torch
import torch.nn as nn


def soft_target_update(source_net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Update the target network parameters using a soft update rule.

    Args:
        source_net: The source network whose parameters are used for updating the target network.
        target_net: The target network whose parameters are updated.
        tau: The interpolation parameter for the soft update rule.
    """
    with torch.no_grad():
        torch._foreach_lerp_(
            list(target_net.parameters()),
            list(source_net.parameters()),
            tau,
        )
