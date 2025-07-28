from pathlib import Path
from typing import Union

from .base_renderer import BaseRenderer
from loguru import logger
import nvdiffrast.torch as dr
import torch


class NvdiffrastRenderer(BaseRenderer, name="nvdiffrast"):
    _background_colors = {
        "black": 0.,
        "white": 1.
    }

    def __init__(self,
                 camera_type: str = "perspective",
                 height: int = None,
                 width: int = None,
                 camera_path: Union[Path, str] = None,
                 blur_radius: int = 7,
                 kernel_size: int = 27,
                 background_color: str = "black",
                 **kwargs
                 ):
        logger.warning("Using Cuda context only for speed. Otherwise use pytorch3D renderer.")
        # check device in kwargs
        if "device" in kwargs:
            if isinstance(kwargs["device"], str):
                device = kwargs["device"]
            elif isinstance(kwargs["device"], torch.device):
                device = kwargs["device"].type
            else:
                raise ValueError(f"Invalid type(device): {type(kwargs['device'])}")
            assert device == "cuda", "Only support cuda device for nvdiffrast renderer."
        else:
            kwargs["device"] = "cuda"
        self.device = kwargs["device"]
        super().__init__(
            camera_type=camera_type,
            height=height,
            width=width,
            camera_path=camera_path,
            blur_radius=blur_radius,
            kernel_size=kernel_size,
            **kwargs
        )
        self.context = dr.RasterizeCudaContext(device=self.device)
        assert background_color.lower() in self._background_colors, f"Background color {background_color} not found. Support only {', '.join(self._background_colors.keys())}"
        self.background_color = background_color.lower()

    def _transform_points(self, transform, verts_padded):
        verts_padded_hom = torch.cat([verts_padded, torch.ones_like(verts_padded[..., :1])], dim=-1)
        transform_matrix = transform.get_matrix().squeeze()
        x_padded_hom = verts_padded_hom @ transform_matrix  # (n_frames, n_verts, 4) @ (4, 4)
        return x_padded_hom

    def _render(
            self,
            verts_padded,
            faces_packed,
            verts_uvs_packed,
            faces_uvs_packed,
            texture,
            antialias=True,
            crop=True,
            filter_mode="linear",
            **kwargs
    ):
        # Transform vertices to NDC space
        x_padded_hom = self._transform_points(self.world_to_ndc_transform, verts_padded)
        # Nvdiffrast requirement: tri, ranges must be int32 tensors
        faces_packed = faces_packed.to(torch.int32)
        faces_uvs_packed = faces_uvs_packed.to(torch.int32)
        # Render
        rast, diff_rast = dr.rasterize(self.context, x_padded_hom, faces_packed, resolution=self.resolution)
        image_attributes, _ = dr.interpolate(verts_uvs_packed[None, ...], rast, faces_uvs_packed, rast_db=diff_rast, diff_attrs=None)
        rgb = dr.texture(texture[None, ...].flip(dims=[1]), uv=image_attributes[..., [0, 1]], filter_mode=filter_mode)
        if antialias:
            rgb = dr.antialias(rgb, rast, x_padded_hom, faces_packed)
        rgb_sil = torch.cat([rgb, torch.ones(*rgb.shape[:-1], 1, device=x_padded_hom.device)], dim=-1)
        if crop:
            # rgb
            rgb_sil = torch.where(rast[..., 3:] > 0, rgb_sil, torch.tensor([self._background_colors[self.background_color]], device=self.device))
            # silhouette
            rgb_sil[..., 3:] = torch.where(rast[..., 3:] > 0, rgb_sil[..., 3:], torch.tensor([0.0], device=self.device))
        return rgb_sil.flip(dims=[1])  # (B, H, W, 4)

    def _render_with_depth(
            self,
            verts_padded,
            faces_packed,
            verts_uvs_packed,
            faces_uvs_packed,
            texture,
            antialias=True,
            crop=True,
            filter_mode="linear",
            **kwargs
    ):
        # Transform vertices to NDC space
        x_padded_hom = self._transform_points(self.world_to_ndc_transform, verts_padded)
        verts_depths_padded = x_padded_hom[..., 2:3].contiguous()
        # Nvdiffrast requirement: tri, ranges must be int32 tensors
        faces_packed = faces_packed.to(torch.int32)
        faces_uvs_packed = faces_uvs_packed.to(torch.int32)
        # Render
        rast, diff_rast = dr.rasterize(self.context, x_padded_hom, faces_packed, resolution=self.resolution)
        pixels_uvs_padded, _ = dr.interpolate(verts_uvs_packed[None, ...], rast, faces_uvs_packed, rast_db=diff_rast,
                                             diff_attrs=None)
        rgb = dr.texture(texture[None, ...].flip(dims=[1]), uv=pixels_uvs_padded[..., [0, 1]], filter_mode=filter_mode)
        depth, _ = dr.interpolate(verts_depths_padded, rast, faces_packed, rast_db=diff_rast, diff_attrs=None)

        if antialias:
            rgb = dr.antialias(rgb, rast, x_padded_hom, faces_packed)

        rgb_sil = torch.cat([rgb, torch.ones(*rgb.shape[:-1], 1, device=x_padded_hom.device)], dim=-1)
        if crop:
            # rgb
            rgb_sil = torch.where(
                rast[..., 3:] > 0, rgb_sil,
                torch.tensor([self._background_colors[self.background_color]], device=self.device)
            )
            # silhouette
            rgb_sil[..., 3:] = torch.where(
                rast[..., 3:] > 0, rgb_sil[..., 3:],
                torch.tensor([0.0], device=self.device)
            )
            # depth
            depth = torch.where(rast[..., 3:] > 0, depth, torch.tensor([-1.0], device=self.device))
        return rgb_sil.flip(dims=[1]), depth.flip(dims=[1])  # (B, H, W, 4) and (B, H, W, 1)
