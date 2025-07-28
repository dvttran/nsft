import json
from pathlib import Path
from typing import Union
from .base_renderer import BaseRenderer
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
    SoftPhongShader,
    AmbientLights,
    BlendParams,
    PointLights,
    TexturesUV,
)
from loguru import logger


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        rgb_sil = self.shader(fragments, meshes_world, **kwargs)
        depth = fragments.zbuf
        return rgb_sil, depth


class Pytorch3dRenderer(BaseRenderer, name="pytorch3d"):
    _background_colors = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
    }
    _shaders = {
        "hard": HardPhongShader,
        "soft": SoftPhongShader
    }
    _rasterizers = {
        "mesh": MeshRasterizer,
    }

    def __init__(self,
                 camera_type: str = "perspective",
                 height: int = None,
                 width: int = None,
                 camera_path: Union[Path, str] = None,
                 blur_radius: int = 7,
                 kernel_size: int = 27,
                 sigma: float = 1e-4,
                 gamma: float = 1e-4,
                 background_color: str = "black",
                 shader: str = "hard",
                 rasterizer: str = "mesh",
                 light_position=None,
                 device: Union[str, torch.device] = "cpu",
                 cull_backfaces: bool = False,
                 **kwargs
                 ):
        super().__init__(
            camera_type=camera_type,
            height=height,
            width=width,
            camera_path=camera_path,
            blur_radius=blur_radius,
            kernel_size=kernel_size,
            device=device,
            **kwargs,
        )
        self.device = device
        self.sigma = sigma
        self.gamma = gamma
        assert background_color.lower() in self._background_colors, f"Background color {background_color} not found. Support only {', '.join(self._background_colors.keys())}"
        self.background_color = background_color.lower()
        self.blend_params = BlendParams(
            sigma,
            gamma,
            background_color=self._background_colors[self.background_color]
        )
        try:
            self.shader = self._shaders[shader]
        except KeyError:
            raise KeyError(f"Shader {shader} not found. Support only {', '.join(self._shaders.keys())}")
        try:
            self.rasterizer = self._rasterizers[rasterizer]
        except KeyError:
            raise KeyError(f"Rasterizer {rasterizer} not found. Support only {', '.join(self._rasterizers.keys())}")

        if light_position:
            lights = PointLights(device=device, location=light_position)
        else:
            lights = AmbientLights(device=device)

        self.renderer = MeshRendererWithDepth(
            rasterizer=self.rasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    image_size=self.resolution,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=cull_backfaces
                )
            ),
            shader=HardPhongShader(
                device=device,
                lights=lights,
                cameras=self.cameras,
                blend_params=self.blend_params
            )
        )
        self.template = None

    def _render(
            self,
            verts_padded,
            faces_packed,
            verts_uvs_packed,
            faces_uvs_packed,
            texture,
            **kwargs
    ):
        tex = TexturesUV(verts_uvs=verts_uvs_packed[None, ...], faces_uvs=faces_uvs_packed[None, ...], maps=texture[None, ...])
        if self.template is None:
            self.template = Meshes(verts=verts_padded[0:1], faces=faces_packed[None, ...], textures=tex)
        meshes = self.template.extend(len(verts_padded)).update_padded(verts_padded)

        rgb_sil = []
        depth = []
        for i in range(0, len(meshes)):
            rgb_sil_i, depth_i = self.renderer(meshes[i])
            rgb_sil.append(rgb_sil_i)
            depth.append(depth_i)
        rgb_sil = torch.cat(rgb_sil, dim=0)
        depth = torch.cat(depth, dim=0)

        return rgb_sil

    def _render_with_depth(
            self,
            verts_padded,
            faces_packed,
            verts_uvs_packed,
            faces_uvs_packed,
            texture,
            **kwargs
    ):
        tex = TexturesUV(verts_uvs=verts_uvs_packed[None, ...], faces_uvs=faces_uvs_packed[None, ...],
                         maps=texture[None, ...])
        if self.template is None:
            self.template = Meshes(verts=verts_padded[0:1], faces=faces_packed[None, ...], textures=tex)
        meshes = self.template.extend(len(verts_padded)).update_padded(verts_padded)

        rgb_sil = []
        depth = []
        for i in range(0, len(meshes)):
            rgb_sil_i, depth_i = self.renderer(meshes[i])
            rgb_sil.append(rgb_sil_i)
            depth.append(depth_i)
        rgb_sil = torch.cat(rgb_sil, dim=0)
        depth = torch.cat(depth, dim=0)

        return rgb_sil, depth
