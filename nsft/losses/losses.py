from typing import Union, List
from .data_losses import LpLoss
from .vision_losses import RGBLoss, SilLoss, ImageDerivativeLoss
from .geometry_losses import MeshInexentsibilityLoss, AdaptiveMeshInexentsibilityLoss
from .metrics import ChamferError, DepthMapRMSE
from .thin_shell_losses import ThinShellLoss


class Losses:
    _losses = {
        "lp": LpLoss,
        "rgb": RGBLoss,
        "sil": SilLoss,
        "image_derivative": ImageDerivativeLoss,
        "mesh_inextensibility": MeshInexentsibilityLoss,
        "adaptive_mesh_inextensibility": AdaptiveMeshInexentsibilityLoss,
        "thin_shell": ThinShellLoss,
        # metrics
        "chamfer": ChamferError,
        "depth_rmse": DepthMapRMSE,
    }

    def __init__(self, losses_config: Union[dict, List[dict]], **kwargs):
        self.losses = []
        self.weights = []
        self.values = []
        self.names = []
        self.render_loss = None
        if isinstance(losses_config, dict):
            losses_config = [losses_config]
        for loss_config in losses_config:
            name = loss_config['name'].lower()
            assert name in self._losses.keys(), f"Loss {name} not found. Supported only {', '.join(self._losses.keys())}"
            if 'hyperparams' in loss_config.keys():
                hyperparams = loss_config['hyperparams']
            else:
                hyperparams = {}
            hyperparams.update(kwargs)
            if 'weight' in loss_config.keys():
                weight = loss_config['weight']
                if weight is None or weight == 0:
                    continue
            else:
                weight = 1.0

            for key, value in self._losses.items():
                if name == key:
                    loss_fn = self._losses[name](**hyperparams)
                    self.losses.append(loss_fn)
                    self.names.append(name.capitalize())

            self.weights.append(weight)

    def __call__(self, warmup: bool = False, **kwargs):
        loss = 0
        self.values = []
        for i, loss_fn in enumerate(self.losses):
            loss_value = loss_fn(**kwargs)
            self.values.append(loss_value.mean().item())
            loss += self.weights[i] * loss_value

        return loss

    def __str__(self):
        return ", ".join([f"{name}: {w*l:.4f}" for (name, w, l) in zip(self.names, self.weights, self.values)])
