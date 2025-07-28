import torch.optim as optim


class Optimizer:
    _optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }

    def __init__(self, name: str):
        super().__init__()
        self.name = name.lower()

    def create(self, parameters, **kwargs):
        for key, value in self._optimizers.items():
            if self.name == key:
                return value(parameters, **kwargs)

        raise ValueError(f"Optimizer '{self.name}' not found. Supported: {', '.join(self._optimizers.keys())}.")


def get_optimizer(parameters, name: str, **kwargs):
    optimizer = Optimizer(name)
    return optimizer.create(parameters, **kwargs)