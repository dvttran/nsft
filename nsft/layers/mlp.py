import torch.nn as nn
from typing import Callable
from loguru import logger


class LinearBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: Callable,
            bias: bool = True
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            activation,
        )

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            n_layers: int,
            activation: Callable,
            final_activation: Callable,
            zero_init: bool = False
    ):
        super().__init__()

        input_layer = [LinearBlock(in_channels, hidden_channels, activation, bias=True)]
        hidden_layers = []
        for _ in range(n_layers - 1):
            hidden_layers.append(LinearBlock(hidden_channels, hidden_channels, activation, bias=True))
        if final_activation is None:
            output_layer = [nn.Linear(hidden_channels, out_channels, bias=True)]
        else:
            output_layer = [nn.Linear(hidden_channels, out_channels, bias=True), final_activation]

        self.layers = nn.ModuleList(input_layer + hidden_layers + output_layer)
        self.layers = nn.Sequential(*self.layers)

        if zero_init:
            logger.info("Initializing the MLP with near-zero parameters...")
            self.apply(self._small_weights_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _small_weights_init(self, module):
        """
        Initialize the weights of the network using Xavier initialization with small gain
        """
        if isinstance(module, nn.Linear):
            # nn.init.xavier_uniform_(module.weight.data, gain=1e-1)
            nn.init.normal_(module.weight.data, mean=0, std=0.001)
            if module.bias is not None:
                # module.bias.data.zero_()
                nn.init.normal_(module.bias.data, mean=0, std=0.001)
