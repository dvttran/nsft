import json
from functools import partial

import yaml
from typing import Union, List, Tuple
from pathlib import Path
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras
)
from pytorch3d.transforms import Transform3d
from loguru import logger
import torchgeometry as tgm
import numpy as np


# Global variables
_R = torch.eye(3)
_T = torch.zeros(1, 3)
_near = 0.01
_far = 1000.
_max_frames_render = 10


class BaseRenderer:
    _renderers = dict()
    _camera_types = ("perspective", "fov_perspective")
    _camera_formats = {
        "PHI_SFT_KINECT": ".json",
        "NRSfM_ECCV2020_KINECT": ".intr",
    }

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseRenderer._renderers[name.lower()] = cls
            cls._name = name
        else:
            BaseRenderer._renderers[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __init__(self,
                 camera_type: str = "perspective",
                 height: int = None,
                 width: int = None,
                 camera_path: Union[Path, str] = None,
                 blur_radius: int = 7,
                 kernel_size: int = 27,
                 **kwargs
                 ):
        self.camera_type = camera_type
        assert camera_type in self._camera_types, f"Camera type {camera_type} not found. Support only {', '.join(self._camera_types)}"
        self.height = height
        self.width = width
        self.blur_radius = blur_radius
        self.kernel_size = kernel_size
        self.gaussian_blur = tgm.image.GaussianBlur((kernel_size, kernel_size), (blur_radius, blur_radius))
        if "kinect" in kwargs.keys():
            self.kinect = kwargs["kinect"]
        else:
            self.kinect = False
        self.world_to_ndc_transform, self.cameras = self.get_camera(camera_path, **kwargs)

    def _get_camera_config(self, camera_path: Union[Path, str], **kwargs):
        if isinstance(camera_path, str):
            camera_path = Path(camera_path)
        # switch extension of camera_path file
        # # Phi-SfT camera setting
        if camera_path.suffix == self._camera_formats['PHI_SFT_KINECT']:
            with open(camera_path, "r") as f:
                camera_config = json.load(f)
        # elif camera_path.suffix == ".yaml":
        #     with open(camera_path, "r") as f:
        #         camera_config = yaml.safe_load(f)

        # # NRSfM ECCV 2020 kinect camera setting
        elif camera_path.suffix == self._camera_formats['NRSfM_ECCV2020_KINECT']:
            with open(camera_path, "r") as f:
                K = np.loadtxt(f)
            camera_config = {
                "fx": K[0, 0],
                "fy": K[1, 1],
                "cx": K[0, 2],
                "cy": K[1, 2],
                "width": int(K[0, 2]) * 2,
                "height": int(K[1, 2]) * 2
            }
        else:
            raise NotImplementedError(f"Camera file extension {camera_path.suffix} not supported.")
        # synchronize config
        camera_config.update(kwargs)

        if self.height:
            camera_config.update(height=self.height)
        elif "height" in camera_config.keys():
            self.height = camera_config["height"]
        else:
            raise ValueError("Image height must be given.")

        if self.width:
            camera_config.update(width=self.width)
        elif "width" in camera_config.keys():
            self.width = camera_config["width"]
        else:
            raise ValueError("Image width must be given.")

        return camera_config

    def get_camera(self, camera_path: Union[Path, str], **kwargs):
        camera_config = self._get_camera_config(camera_path, **kwargs)

        # load camera from config
        if self.camera_type == "perspective":
            return self._get_perspective_camera(**camera_config)
        elif self.camera_type == "fov_perspective":
            return self._get_fov_perspective_camera(**camera_config)
        else:
            raise ValueError(f"Got camera type={self.camera_type}, expected one of {self._camera_types}.")

    def _get_fov_perspective_camera(self, object_pos, camera_pos,
                                    up_direction=(0, 1., 0),
                                    fov: float = 60.,
                                    aspect_ratio: float = 1.,
                                    near: float = _near,
                                    far: float = _far,
                                    device: Union[torch.device, str] = "cpu",
                                    **kwargs
                                    ):
        R, T = look_at_view_transform(eye=(camera_pos,), up=(up_direction,), at=(object_pos,), device=device)
        cameras = FoVPerspectiveCameras(
            znear=near,
            zfar=far,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            device=device
        )
        world_to_ndc_transform = cameras.get_full_projection_transform()
        # world_to_ndc_transform = Transform3d(
        #     matrix=torch.tensor([
        #         [-1., 0, 0, 0],
        #         [0, 1., 0, 0],
        #         [0, 0, 1., 0],
        #         [0, 0, 0, 1.]
        #     ], device=device, dtype=world_to_ndc_transform.dtype).transpose(-1, -2)
        # ).compose(world_to_ndc_transform)
        world_to_ndc_transform = world_to_ndc_transform.compose(Transform3d(
            matrix=torch.tensor([
                [-1., 0, 0, 0],
                [0, 1., 0, 0],
                [0, 0, 1., 0],
                [0, 0, 0, 1.]
            ], device=device, dtype=world_to_ndc_transform.dtype).transpose(-1, -2)
        ))
        # world_to_ndc_transform = self._convert_pytorch3d_to_opengl(cameras.get_full_projection_transform())

        # world_to_view_transform = self._convert_pytorch3d_to_opengl(cameras.get_world_to_view_transform())
        # view_to_ndc_transform = cameras.get_projection_transform()
        # world_to_ndc_transform = world_to_view_transform.compose(view_to_ndc_transform)

        return world_to_ndc_transform, cameras

    def _get_perspective_camera(self, fx, fy, cx, cy, height, width,
                                near: float = _near,
                                far: float = _far,
                                R: Union[List, Tuple, torch.Tensor, np.array] = _R,
                                T: Union[List, Tuple, torch.Tensor, np.array] = _T,
                                device: Union[torch.device, str] = "cpu",
                                dtype: torch.dtype = torch.float32,
                                **kwargs
                                ):
        """
        intrinsic matrix:
             K =
             [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]]
        camera pose:
            R, T
        """
        logger.warning("The intrinsic parameters must be given in screen coordinates !")
        logger.warning("`R` and `T` follow Multi-View Geometry convention !")

        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, device=device, dtype=dtype)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, device=device, dtype=dtype)
        R = R.to(device=device, dtype=dtype)
        T = T.to(device=device, dtype=dtype)

        if R.ndim == 2:
            R = R[None, ...]
        if T.ndim == 1:
            T = T[None, ...]

        if self.kinect:
            R_kinect = torch.tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]], device=device, dtype=dtype)
            R = R_kinect @ R
            T = (R_kinect @ T.unsqueeze(-1)).squeeze(-1)

        cameras = PerspectiveCameras(
            focal_length=((fx, fy), ),
            principal_point=((cx, cy), ),
            R=R.transpose(-1, -2),  # Pytorch3d convention R.transpose(-1, -2)
            T=T,
            in_ndc=False,
            image_size=((height, width), ),
            device=device
        )

        world_to_view_transform = self._convert_pytorch3d_to_opengl(cameras.get_world_to_view_transform(), dtype=dtype)
        # if self.kinect:
        #     world_to_view_transform = world_to_view_transform.compose(Transform3d(
        #         matrix=torch.tensor([
        #             [-1., 0, 0, 0],
        #             [0, -1., 0, 0],
        #             [0, 0, 1., 0],
        #             [0, 0, 0, 1.]
        #         ], device=device, dtype=dtype).transpose(-1, -2)
        #     ))
        view_to_ndc_transform = self._get_view_to_ndc_transform(fx, fy, cx, cy, height, width, near, far, device, dtype)
        world_to_ndc_transform = world_to_view_transform.compose(view_to_ndc_transform)

        logger.warning("`world_to_ndc_transform` follows OpenGL convention !")
        return world_to_ndc_transform, cameras

    def _convert_pytorch3d_to_opengl(self, world_to_view_transform: Transform3d, dtype: torch.dtype = torch.float32):
        logger.warning("`_convert_pytorch3d_to_opengl` still requires flipping texture and render images.")
        world_to_view_matrix = world_to_view_transform.get_matrix().squeeze()
        # return world_to_view_transform
        return world_to_view_transform.compose(Transform3d(
            matrix=torch.tensor([
                [-1., 0, 0, 0],
                [0, 1., 0, 0],
                [0, 0, -1., 0],
                [0, 0, 0, 1.]
            ], device=world_to_view_matrix.device, dtype=dtype).transpose(-1, -2)
        ))

    def _get_view_to_ndc_transform(self, fx, fy, cx, cy, height, width, near, far,
                                   device: Union[torch.device, str] = "cpu",
                                   dtype: torch.dtype = torch.float32):
        A = (2 * fx) / width
        B = (2 * fy) / height
        C = (width - 2 * cx) / width
        D = -(height - 2 * cy) / height
        E = (near + far) / (near - far)
        F = (2 * near * far) / (near - far)

        view_to_ndc = torch.tensor([
            [A, 0, C, 0],
            [0, B, D, 0],
            [0, 0, E, F],
            [0, 0, -1., 0]
        ], device=device, dtype=dtype)

        view_to_ndc_transform = Transform3d(
            matrix=view_to_ndc.transpose(-1, -2).contiguous(), device=device, dtype=dtype
        )  # Pytorch3D convention uses transposed matrix for batch multiplication

        return view_to_ndc_transform

    def _render(*args, **kwargs):
        pass

    def render(self,
               verts_padded,
               faces_packed,
               verts_uvs_packed,
               faces_uvs_packed,
               texture,
               max_frames_render: int = _max_frames_render,
               render_depth: bool = False,
               **kwargs
               ):
        n_meshes = len(verts_padded)
        if n_meshes > max_frames_render:
            n_chunks = np.ceil(n_meshes / max_frames_render).astype(int)
        else:
            n_chunks = 1
        chunks = torch.chunk(torch.arange(n_meshes), n_chunks, dim=0)

        if render_depth:
            rgb_sil = []
            depth = []
            for chunk in chunks:
                rgb_sil_chunk, depth_chunk = self._render_with_depth(
                    verts_padded=verts_padded[chunk],
                    faces_packed=faces_packed,
                    verts_uvs_packed=verts_uvs_packed,
                    faces_uvs_packed=faces_uvs_packed,
                    texture=texture,
                    **kwargs
                )
                rgb_sil.append(rgb_sil_chunk)
                depth.append(depth_chunk)
            rgb_sil = torch.cat(rgb_sil, dim=0)
            depth = torch.cat(depth, dim=0)
            rgb = rgb_sil[..., :3]
            sil = rgb_sil[..., 3:]
        else:
            rgb_sil = []
            for chunk in chunks:
                rgb_sil_chunk = self._render(
                    verts_padded=verts_padded[chunk],
                    faces_packed=faces_packed,
                    verts_uvs_packed=verts_uvs_packed,
                    faces_uvs_packed=faces_uvs_packed,
                    texture=texture,
                    **kwargs
                )
                rgb_sil.append(rgb_sil_chunk)
            rgb_sil = torch.cat(rgb_sil, dim=0)
            rgb = rgb_sil[..., :3]
            sil = rgb_sil[..., 3:]

        mask = sil > 0.0

        sil = torch.transpose(sil, -1, -3)
        sil = self.gaussian_blur(sil)
        sil = torch.transpose(sil, -1, -3)

        if render_depth:
            return rgb, sil, depth, mask
        else:
            return rgb, sil

    @property
    def resolution(self):
        return self.height, self.width


def available_renderers():
    return list(BaseRenderer._renderers.keys())


def get_render_fn_and_cameras(renderer_config: dict):
    name = renderer_config.pop("name").lower()
    try:
        camera_path = renderer_config.pop("camera_path")
    except KeyError:
        raise ValueError("Camera path must be given.")

    try:
        renderer = BaseRenderer._renderers[name](camera_path=camera_path, **renderer_config)
        # TODO: check renderer.world_to_ndc_transform.get_matrix()
        render_fn = lambda verts_padded, faces_packed, verts_uvs_packed, faces_uvs_packed, texture, render_depth: renderer.render(
            verts_padded=verts_padded,
            faces_packed=faces_packed,
            verts_uvs_packed=verts_uvs_packed,
            faces_uvs_packed=faces_uvs_packed,
            texture=texture,
            render_depth=render_depth,
            **renderer_config
        )
        cameras = renderer.cameras
        return render_fn, cameras

    except KeyError:
        raise ValueError(f"Got renderer name={name}, expected one of {available_renderers()}.")
