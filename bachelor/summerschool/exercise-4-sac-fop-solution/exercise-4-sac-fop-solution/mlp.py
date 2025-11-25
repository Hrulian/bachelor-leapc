import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)

class Mlp(nn.Module):
    """A base class for a multi-layer perceptron (MLP) 
    with 3 hidden layers of size 256 and ReLU activations.

    Attributes:
        mlp: The multi-layer perceptron model.
    """

    mlp: nn.Module

    def __init__(
        self,
        input_sizes: int | list[int],
        output_sizes: int | list[int],
    ) -> None:
        """Initializes the MLP.

        Args:
            input_sizes: The sizes of the input tensors. Inputs will be concatenated.
            output_sizes: The sizes of the output tensors.
                Outputs will be split according to these sizes.
        """
        super().__init__()

        self.activation = nn.ReLU()
        hidden_dims = (256, 256, 256)

        if isinstance(input_sizes, int):
            input_sizes = [input_sizes]
        self._comb_input_dim = sum(input_sizes)
        self._input_dims = input_sizes

        if isinstance(output_sizes, int):
            output_sizes = [output_sizes]
        self._comb_output_dim = sum(output_sizes)
        self._output_dims = output_sizes

        # mlp
        layers = []
        prev_d = self._comb_input_dim
        for d in [*hidden_dims, self._comb_output_dim]:
            layers.extend([nn.Linear(prev_d, d), self.activation])
            prev_d = d

        self.mlp = nn.Sequential(*layers[:-1])

        # Apply weight initialization to this MLP
        self.mlp.apply(orthogonal_init)

    def forward(self, *x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=-1)  # type: ignore
        y = self.mlp(x)  # type: ignore

        if len(self._output_dims) == 1:
            return y

        return torch.split(y, self._output_dims, dim=-1)
