import torch
if torch.cuda.is_available():
    from .nvdiffrast_renderer import NvdiffrastRenderer
from .pytorch3d_renderer import Pytorch3dRenderer
from .base_renderer import get_render_fn_and_cameras
