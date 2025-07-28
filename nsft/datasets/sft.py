import os
from functools import partial
from pathlib import Path
from typing import Union, List
import numpy as np
import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from .base_dataset import BaseDataset
from torch.utils.data import Dataset
import imageio.v3 as iio
from loguru import logger
from ..renderers import get_render_fn_and_cameras
from ..surfaces import get_surface
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
import json
from matplotlib import pyplot as plt
import trimesh
from scipy.io import loadmat
from matplotlib import cm


class SfTDataset(BaseDataset, Dataset, name="sft"):
    _trajectories = ("coupled", "decoupled")
    start_time: float = 0.
    end_time: float

    def __init__(self,
                 root_dir: str,
                 dataset_name: str = '',
                 sequence_type: str = '',
                 sequence_name: str = '',
                 rgb_file: Union[Path, str] = None,
                 rgb_dir: Union[Path, str] = None,
                 sil_file: Union[Path, str] = None,
                 sil_dir: Union[Path, str] = None,
                 mask_file: Union[Path, str] = None,
                 mask_dir: Union[Path, str] = None,
                 gt_mesh_dir: Union[Path, str] = None,
                 gt_point_cloud_dir: Union[Path, str] = None,
                 gt_kinect_data_dir: Union[Path, str] = None,
                 gt_aligned: bool = True,
                 trajectory_type: str = "decoupled",
                 dt: float = None,
                 device: Union[torch.device, str] = "cpu",
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ):
        super().__init__(root_dir, dataset_name, sequence_type, sequence_name, **kwargs)
        self.device = device

        # render
        render_fn, cam_params = self.get_render_fn(device=device, dtype=dtype, **kwargs)
        self.cam_params = cam_params

        # template
        template = self.get_template(cam_params, device=device, dtype=dtype, **kwargs)
        self.mesh = template
        self.render_fn = partial(render_fn,
            faces_packed=template.faces_packed,
            verts_uvs_packed=template.verts_uvs_packed,
            faces_uvs_packed=template.faces_uvs_packed,
            texture=template.texture,
            render_depth=False,
        )
        self.render_with_depth_fn = partial(render_fn,
            faces_packed=template.faces_packed,
            verts_uvs_packed=template.verts_uvs_packed,
            faces_uvs_packed=template.faces_uvs_packed,
            texture=template.texture,
            render_depth=True,
        )
        self.template = Meshes(
            verts=[template.verts_packed],
            faces=[template.faces_packed],
            textures=TexturesUV(
                verts_uvs=template.verts_uvs_packed[None],
                faces_uvs=template.faces_uvs_packed[None],
                maps=template.texture[None]
            )
        )
        self.texture = template.texture
        self.eig_vals = template.eig_vals
        self.adj_mat = template.adj_mat
        self.edges_packed = template.E
        self.edge_lengths_0 = template.edge_lengths_packed
        self.faces_areas_0 = template.faces_areas_packed
        self.get_faces_areas_padded = lambda verts_padded: template.get_faces_areas_padded(
            verts_padded=verts_padded,
            faces_packed=template.faces_packed
        )

        # ground-truth data
        rgb, sil, mask, gt_meshes, gt_point_clouds, gt_point_clouds_lengths, gt_kinect_data = self.read_data(
            rgb_file,
            rgb_dir,
            sil_file,
            sil_dir,
            mask_file,
            mask_dir,
            gt_mesh_dir,
            gt_point_cloud_dir,
            gt_kinect_data_dir,
            **kwargs)
        self.gt_aligned = gt_aligned

        # images/video
        if rgb is not None:
            self.target_rgb = torch.from_numpy(rgb[..., :3]).to(dtype=dtype, device=device) / 255.
            self.n_time_span = len(rgb)
        if sil is not None:
            self.target_sil = torch.from_numpy(sil[..., :1]).to(dtype=dtype, device=device) / 255.
            self.n_time_span = len(sil)
        if mask is not None:
            self.gt_mask = (torch.from_numpy(mask[..., :1]).to(dtype=dtype, device=device) / 255.) > 0.0
        # ground-truth
        if gt_meshes is not None:
            self.n_time_span = len(gt_meshes)
            try:
                self.x = gt_meshes.verts_padded().to(dtype=dtype, device=device)
            except Exception as e:
                logger.error("Error loading ground-truth meshes: ", e)
                self.x = None
        else:
            self.x = None
        if gt_point_clouds is not None:
            self.gt_point_clouds = gt_point_clouds
            self.gt_point_clouds_lengths = torch.tensor(gt_point_clouds_lengths, device=device)
        else:
            self.gt_point_clouds = None
            self.gt_point_clouds_lengths = None
        if gt_kinect_data is not None:
            self.gt_kinect_data = torch.from_numpy(gt_kinect_data).to(device=device)
        else:
            self.gt_kinect_data = None
        # trajectory
        self.n_vertices, self.data_dim = template.verts_packed.shape
        if dt:
            self.end_time = self.start_time + self.n_time_span * dt
            self.time_span = torch.arange(self.start_time, self.end_time, dt, device=device)
        else:
            self.end_time = 1.
            self.time_span = torch.linspace(self.start_time, self.end_time, self.n_time_span, device=device)

        # Parse trajectory type
        if self.x is not None:
            self.x0 = self.x[0].clone()
        else:
            self.x0 = template.verts_packed.clone()
        self._parse_trajectory(trajectory_type, **kwargs)

    def _parse_trajectory(self, trajectory_type: str, **kwargs):
        assert trajectory_type.lower() in self._trajectories, f"Trajectory type {trajectory_type} not found. Support only {', '.join(self._trajectories)}"
        trajectory_type = trajectory_type.lower()
        self.trajectory_type = trajectory_type
        if trajectory_type == "coupled":
            self.x0 = self.x0.flatten(-2, -1)[None]  # (1, data_channels)
            if self.x is not None:
                self.x = self.x.flatten(-2, -1)[:, None]  # (n_time_span, 1, data_channels)
            self.data_channels = self.n_vertices * self.data_dim
            self.dataset_size = 1
        elif trajectory_type == "decoupled":
            self.x0 = self.x0  # (n_vertices, data_dim)
            if self.x is not None:
                self.x = self.x
            self.data_channels = self.data_dim
            self.dataset_size = self.n_vertices
        else:
            raise ValueError(
                f"Trajectory type {trajectory_type} not found. Support only {', '.join(self._trajectories)}")

    def get_render_fn(self, device: Union[torch.device, str], dtype: torch.dtype, **kwargs):
        cam_params = {}
        if "render" in kwargs:
            render_config = kwargs["render"]
            render_config.update({"device": device, "dtype": dtype})
            if "kinect" in render_config:
                cam_params.update({"kinect": render_config["kinect"]})
            if "image_size" in render_config:
                render_config.update(
                    {"width": render_config["image_size"]["width"],
                     "height": render_config["image_size"]["height"]})
            # Parse camera config
            if "camera_file" in render_config:
                camera_file = render_config.pop("camera_file")
                camera_path = Path(self.data_dir).joinpath(camera_file)
                with open(camera_path, "r") as f:
                    calibration = json.load(f)
                    if 'fx' in calibration:  # perspective
                        K = torch.tensor([
                            [calibration["fx"], 0, calibration["cx"]],
                            [0, calibration["fy"], calibration["cy"]],
                            [0, 0, 1]
                        ], device=device, dtype=dtype)
                        cam_params.update({"K": K})
                render_config["camera_path"] = camera_path
                if "R" in render_config:
                    R = torch.tensor(render_config["R"], device=device, dtype=dtype)
                else:
                    R = torch.eye(3, device=device, dtype=dtype)
                if "T" in render_config:
                    T = torch.tensor(render_config["T"], device=device, dtype=dtype)
                else:
                    T = torch.zeros(3, device=device, dtype=dtype)
                cam_params.update({"R": R, "T": T})
            elif "camera_dir" in render_config:
                logger.warning("`camera_dir` contains `cam.ext` and `cam.intr` files")
                # read files info from camera_dir
                camera_dir = render_config.pop("camera_dir")
                camera_dir = Path(self.data_dir).joinpath(camera_dir)
                cam_ext = Path(camera_dir).joinpath("cam.ext")
                cam_intr = Path(camera_dir).joinpath("cam.intr")

                # extract [R | T] from cam_ext
                with open(cam_ext, "r") as f:
                    RT = np.loadtxt(f)
                R = RT[..., :3]
                T = RT[..., 3]
                cam_params.update({
                    "R": torch.tensor(R, device=device, dtype=dtype),
                    "T": torch.tensor(T, device=device, dtype=dtype)
                })
                render_config.update({"R": R, "T": T})

                # pass `cam_intr` as `camera_path`
                with open(cam_intr, "r") as f:
                    K = np.loadtxt(f)
                cam_params.update({"K": torch.tensor(K, device=device, dtype=dtype)})
                render_config.update({"camera_path": cam_intr})
            else:
                raise ValueError("You must provide `camera_file` or `camera_dir` in render config")
            # Get render
            render_fn, cameras = get_render_fn_and_cameras(render_config)
            cam_params.update({"cameras": cameras})
            return render_fn, cam_params
        else:
            raise ValueError("You must provide render config")

    def get_template(self, cam_params, device: Union[torch.device, str], dtype: torch.dtype, **kwargs):
        if "template" in kwargs:
            template_config = kwargs["template"]
            template_config.update({"device": device, "dtype": dtype})
            # 3D template
            if "template_file" in template_config:
                template_file = template_config.pop("template_file")
                template_path = Path(self.data_dir).joinpath(template_file)
                template_config.update({"surface_path": template_path})
            elif "vertex_file" and "face_file" in template_config:
                vertex_file = template_config.pop("vertex_file")
                face_file = template_config.pop("face_file")
                vertex_path = Path(self.data_dir).joinpath(vertex_file)
                face_path = Path(self.data_dir).joinpath(face_file)
                template_config.update({"vertex_path": vertex_path, "face_path": face_path})
            else:
                raise ValueError("You must provide template files")
            # Texture
            if "texture_file" in template_config:
                texture_file = template_config.pop("texture_file")
                texture_path = Path(self.data_dir).joinpath(texture_file)
                template_config.update({"texture_path": texture_path})
            else:
                raise ValueError("You must provide texture file")
            template_config.update(cam_params)
            template = get_surface(surface_config=template_config)
            return template
        else:
            raise ValueError("You must provide template config")

    def get_vertices(self, x):
        # if coupled, x: (n_time_span, data_channels)
        # elif decoupled, x: (n_time_span, n_vertices, data_dim)
        if self.trajectory_type == "coupled":
            return x.view(-1, self.n_vertices, self.data_dim)
        elif self.trajectory_type == "decoupled":
            if x.ndim == 2:
                return x[None, ...]
            return x

    def align_to_gt(self, verts_padded):
        if self.gt_aligned:
            return verts_padded
        else:
            R = self.cam_params["R"]
            T = self.cam_params["T"]
            return verts_padded @ R.T + T

    def inverse_align_to_gt(self, verts_padded):
        R = self.cam_params["R"]
        T = self.cam_params["T"]
        return (verts_padded - T) @ R

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # time_span does not have batch dimension
        # NOTE: We remove rgb & sil that causes memory error
        ground_truth = {
            "x": self.x[:, idx],
            # "target_rgb": self.target_rgb,
            # "target_sil": self.target_sil,
        }
        return self.x0[idx], self.time_span, ground_truth

    def _joinpath(self, *paths):
        return [Path(self.data_dir).joinpath(p) if p else None for p in paths]

    def _read_data(self,
                   rgb_file: Union[Path, str] = None,
                   rgb_dir: Union[Path, str] = None,
                   sil_file: Union[Path, str] = None,
                   sil_dir: Union[Path, str] = None,
                   mask_file: Union[Path, str] = None,
                   mask_dir: Union[Path, str] = None,
                   gt_mesh_dir: Union[Path, str] = None,
                   gt_point_cloud_dir: Union[Path, str] = None,
                   gt_kinect_data_dir: Union[Path, str] = None,
                   **kwargs
                   ):
        # Join path to data directory
        rgb_path, rgb_dir, sil_path, sil_dir, mask_path, mask_dir, gt_mesh_dir, gt_point_cloud_dir, gt_kinect_data_dir = self._joinpath(
            rgb_file, rgb_dir, sil_file, sil_dir, mask_file, mask_dir, gt_mesh_dir, gt_point_cloud_dir, gt_kinect_data_dir
        )

        # Read rgb images/video
        rgb = self._read_video(rgb_path, rgb_dir, **kwargs)
        if rgb is not None and rgb.ndim == 3:  # single frame
            rgb = rgb[None]

        # Read silhouette
        sil = self._read_video(sil_path, sil_dir, **kwargs)
        if sil is not None and sil.ndim == 3:
            if sil.shape[-1] == 1:
                sil = sil[None]  # single frame
            else:
                sil = sil[..., None]  # single channel

        # Read mask
        mask = self._read_video(mask_path, mask_dir, **kwargs)
        if mask is not None and mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[None]
            else:
                mask = mask[..., None]

        # Read ground-truth trajectory: point clouds / meshes
        if gt_mesh_dir:
            gt_meshes = self._read_meshes_from_dir(gt_mesh_dir, **kwargs)
        else:
            gt_meshes = None
        if gt_point_cloud_dir:
            gt_point_clouds, gt_point_clouds_lengths = self._read_point_cloud(gt_point_cloud_dir, **kwargs)
        else:
            gt_point_clouds = None
            gt_point_clouds_lengths = None
        if gt_kinect_data_dir:
            gt_kinect_data = self._read_kinect_data(gt_kinect_data_dir, **kwargs)
        else:
            gt_kinect_data = None

        return rgb, sil, mask, gt_meshes, gt_point_clouds, gt_point_clouds_lengths, gt_kinect_data

    def _read_meshes_from_dir(self, meshes_dir: Union[Path, str], ext=".obj", **kwargs):
        if isinstance(meshes_dir, str):
            meshes_dir = Path(meshes_dir)
        meshes = load_objs_as_meshes([file for file in sorted(meshes_dir.iterdir()) if file.suffix.endswith(ext)], device=self.device)
        return meshes

    def _read_video(self, video_path: Union[Path, str] = None, images_dir: Union[Path, str] = None, **kwargs):
        if video_path:
            return self._read_video_from_path(video_path, **kwargs)
        elif images_dir:
            return self._read_video_from_dir(images_dir, **kwargs)
        else:
            # raise ValueError("You must provide video or images directory")
            logger.warning("You must provide video or images directory")

    def _read_kinect_data(self, kinect_data_dir: Union[Path, str], ext=".mat", key_label="XYZ", **kwargs):
        if isinstance(kinect_data_dir, str):
            kinect_data_dir = Path(kinect_data_dir)
        kinect_data = []
        for file in sorted(kinect_data_dir.iterdir()):
            if file.suffix.endswith(ext):
                data = loadmat(file)[key_label]
                kinect_data.append(data)
        kinect_data = np.stack(kinect_data)
        return kinect_data

    def _read_video_from_path(self, video_path: Union[Path, str], **kwargs):
        try:
            reader = iio.imiter(video_path)
        except Exception as e:
            print("Error opening video file: ", e)
            return None
        frames = []
        for i, im in enumerate(reader):
            frames.append(np.array(im))
        return np.stack(frames)

    def _read_video_from_dir(self, images_dir: Union[Path, str], ext=".png", **kwargs):
        if isinstance(images_dir, str):
            images_dir = Path(images_dir)
        images = list()
        for file in sorted(images_dir.iterdir()):
            if file.suffix.endswith(ext):
                images.append(iio.imread(file))
        return np.stack(images)

    def _read_point_cloud_from_dir(self,
        point_cloud_dir: Union[Path, str],
        point_cloud_ext: str = ".obj",
        point_cloud_delimiter: str = ';',
        **kwargs
    ):
        logger.warning("Point cloud directory should only contain point cloud files")
        point_clouds = []
        point_clouds_lengths = []
        for file in sorted(point_cloud_dir.iterdir()):
            if file.suffix.endswith(point_cloud_ext):
                if point_cloud_ext == ".obj":
                    point_cloud, _, _ = load_obj(file, device=self.device)
                elif point_cloud_ext == ".txt":
                    point_cloud = np.loadtxt(file, delimiter=point_cloud_delimiter)
                    point_cloud = torch.from_numpy(point_cloud).to(device=self.device)
                else:
                    raise NotImplementedError(f"Extension {point_cloud_ext} not supported")
                point_clouds.append(point_cloud)
                point_clouds_lengths.append(len(point_cloud))
        return point_clouds, point_clouds_lengths

    def _read_point_cloud(self, point_cloud_dir: Union[Path, str], **kwargs):
        _point_clouds, point_cloud_lengths = self._read_point_cloud_from_dir(point_cloud_dir, **kwargs)
        max_n_points = max(point_cloud_lengths)
        n_time_span = len(_point_clouds)
        point_clouds = torch.zeros(n_time_span, max_n_points, _point_clouds[0].shape[-1], device=self.device)
        for i, point_cloud in enumerate(_point_clouds):
            point_clouds[i, :point_cloud_lengths[i]] = point_cloud.to(device=self.device)
        return point_clouds, point_cloud_lengths

    def _save_image(self, uri, image, **kwargs):
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
        image = (image * 255).astype(np.uint8)
        iio.imwrite(uri, image, **kwargs)

    def save_data(
            self,
            verts_padded,
            output_path: Union[Path, str],
            surfaces_name: str = '',
            rgbs_name: str = '',
            blurred_masks_name: str = None,
            depths_name: str = None,
            materials: bool = False,
            MIN_DEPTH: float = 400,
            MAX_DEPTH: float = 900,
            **kwargs):
        if isinstance(output_path, str):
            output_path = Path(output_path)

        output_path = output_path.joinpath(self.dataset_name, self.sequence_type, self.sequence_name)

        Path.mkdir(output_path, parents=True, exist_ok=True)
        logger.info(f"Saving outputs to {output_path}...")

        # verts_padded = self.get_vertices(x_pred)
        # render_rgb, render_sil = self.render_fn(verts_padded)
        render_rgb, render_sil, render_depth, render_mask = self.render_with_depth_fn(verts_padded)

        # verts_padded = verts_padded.detach().cpu().numpy()
        render_rgb = render_rgb.detach().cpu().numpy()
        render_sil = render_sil.detach().cpu().numpy()
        render_depth = render_depth.detach().cpu().numpy()
        render_mask = render_mask.detach().cpu().numpy()

        # save rgbs
        if rgbs_name is not None:
            rgb_path = Path(output_path).joinpath("rgbs")
            Path.mkdir(rgb_path, parents=True, exist_ok=True)
            for idx, rgb in enumerate(render_rgb):
                self._save_image(os.path.join(rgb_path, f"{rgbs_name + '_' if rgbs_name else ''}{idx:03d}.png"), rgb)

        # save blurred masks
        if blurred_masks_name is not None:
            mask_path = Path(output_path).joinpath("blurred_masks")
            Path.mkdir(mask_path, parents=True, exist_ok=True)
            for idx, mask in enumerate(render_sil):
                self._save_image(os.path.join(mask_path, f"{blurred_masks_name + '_' if blurred_masks_name else ''}{idx:03d}.png"), mask)

        # save depth maps
        if depths_name is not None:
            depth_path = Path(output_path).joinpath("depths")
            Path.mkdir(depth_path, parents=True, exist_ok=True)
            for idx, (depth, mask) in enumerate(zip(render_depth, render_mask)):
                depth = (render_depth.squeeze() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
                depth = cm.jet(depth)
                depth = (depth * 255).astype("uint8")
                depth[~mask] = 255
                iio.imwrite(os.path.join(depth_path, f"{depths_name + '_' if depths_name else ''}{idx:03d}.png"), depth)

        # save meshes
        verts_padded = self.align_to_gt(verts_padded)
        if surfaces_name is not None:
            mesh_path = Path(output_path).joinpath("surfaces")
            Path.mkdir(mesh_path, parents=True, exist_ok=True)
            mesh = self.template.clone()
            if materials:
                texture_map = self.texture.clone()
            else:
                texture_map = None
            for idx, verts_packed in enumerate(verts_padded):
                mesh = mesh.update_padded(verts_packed[None])
                save_obj(f=os.path.join(mesh_path, f"{idx:04d}{'_' + surfaces_name if surfaces_name else ''}.obj"),
                         verts=mesh.verts_packed(),
                         faces=mesh.faces_packed(),
                         verts_uvs=mesh.textures._verts_uvs_padded.squeeze(),
                         faces_uvs=mesh.textures._faces_uvs_padded.squeeze(),
                         texture_map=texture_map
                )

    def read_output_meshes(self, output_path, dir_name="surfaces", **kwargs):
        if isinstance(output_path, str):
            output_path = Path(output_path)
        logger.info(f"Reading output meshes from {output_path}...")
        meshes_dir = output_path.joinpath(self.dataset_name, self.sequence_type, self.sequence_name, dir_name)

        meshes = self._read_meshes_from_dir(meshes_dir=meshes_dir, **kwargs)
        return meshes

    def read_vertex_colors(self, output_path, dir_name="eval_meshes", ext=".obj", **kwargs):
        if isinstance(output_path, str):
            output_path = Path(output_path)
        meshes_dir = output_path.joinpath(self.dataset_name, self.sequence_type, self.sequence_name, dir_name)

        files = [file for file in sorted(meshes_dir.iterdir()) if file.suffix.endswith(ext)]

        vertex_colors = []
        for file in files:
            mesh = trimesh.load(file)
            colors = mesh.visual.vertex_colors[:, :3]  # Get RGB colors
            vertex_colors.append(colors)
        return np.array(vertex_colors)

    def _save_obj_with_vertex_colors(
            self,
            f: Union[Path, str],
            verts: torch.Tensor,
            faces: torch.Tensor,
            verts_colors: np.array,
            **kwargs
    ):
        if isinstance(f, str):
            f = Path(f)

        if not f.suffix == '.obj':
            raise ValueError(f"File {f} must have .obj extension")

        verts = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()

        with open(f, 'w') as file:
            for v, c in zip(verts, verts_colors):
                file.write(f"v {' '.join(map(str, v))} {' '.join(map(str, c))}\n")
            for face in faces:
                file.write(f"f {' '.join(map(lambda x: str(int(x + 1)), face))}\n")

    def _convert_scalar_field_to_color(
            self,
            scalar_field: torch.Tensor,
            colormap: str = 'RdBu_r',
            **kwargs
    ):
        scalar_field = scalar_field.detach().cpu().numpy()
        scalar_field = (scalar_field + 1) / 2  # Normalize to [0, 1] range
        colormap = plt.get_cmap(colormap)
        colors = colormap(scalar_field)

        return colors[..., :3]

    def save_eval_meshes(
            self,
            verts_padded: torch.Tensor,
            faces_packed: torch.Tensor,
            eval_scalar_field_padded: torch.Tensor,
            output_path: Union[Path, str],
            surfaces_name: str = 'eval',
            **kwargs
    ):
        if isinstance(output_path, str):
            output_path = Path(output_path)
        output_path = output_path.joinpath(self.dataset_name, self.sequence_type, self.sequence_name)
        meshes_dir = output_path.joinpath("eval_meshes")
        logger.info(f"Saving eval meshes to {meshes_dir}...")
        Path.mkdir(meshes_dir, parents=True, exist_ok=True)

        for idx, (verts_packed, eval_scalar_field) in enumerate(zip(verts_padded, eval_scalar_field_padded)):
            mesh_name = os.path.join(meshes_dir, f"{idx:04d}{'_' + surfaces_name if surfaces_name else ''}.obj")
            verts_colors = self._convert_scalar_field_to_color(eval_scalar_field, **kwargs)
            self._save_obj_with_vertex_colors(
                f=mesh_name,
                verts=verts_packed,
                faces=faces_packed,
                verts_colors=verts_colors,
                **kwargs
            )

    def save_images(
            self,
            images_dir,
            images,
            images_name: str = '',
            **kwargs
    ):
        Path.mkdir(images_dir, parents=True, exist_ok=True)
        for idx, image in enumerate(images):
            self._save_image(
                os.path.join(images_dir, f"{images_name + '_' if images_name else ''}{idx:03d}.png"),
                image,
                **kwargs
            )