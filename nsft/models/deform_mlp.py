from ..layers import MLP
from .base_model import BaseModel
from typing import Callable


class DeformMLP(BaseModel, name='DeformMLP'):
    def __init__(
            self,
            data_channels: int,
            hidden_channels: int,
            n_layers: int,
            activation: Callable,
            final_activation: Callable,
            **kwargs
    ):
        super().__init__()
        self.mlp = MLP(
            in_channels=data_channels,
            out_channels=data_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            activation=activation,
            final_activation=final_activation,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.mlp(*args, **kwargs)
