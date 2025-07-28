import random
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import cv2
from torch.func import vmap


class ChamferError:
    def __init__(self,
                 seed: int = None,
                 scale: float = 1e4,
                 **kwargs
                 ):
        self.seed = seed
        self.scale = scale

    def _sample_point_clouds(self, verts_packed, faces_packed, n_samples: int):
        device = verts_packed.device

        triangle_vertices = verts_packed[faces_packed]
        edges = triangle_vertices[:, 1:] - triangle_vertices[:, 0].unsqueeze(1)
        areas = torch.linalg.norm(torch.linalg.cross(edges[:, 0], edges[:, 1], dim=1), dim=1)
        triangle_samples = torch.multinomial(input=areas, num_samples=n_samples, replacement=True)

        uv = torch.rand([n_samples, 2], device=device)
        u = 1 - torch.sqrt(uv[:, 0])
        v = torch.sqrt(uv[:, 0]) * (1 - uv[:, 1])
        w = uv[:, 1] * torch.sqrt(uv[:, 0])
        uvw = torch.stack([u, v, w], dim=1).unsqueeze(2)

        points = torch.sum(triangle_vertices[triangle_samples] * uvw, dim=1)
        return points

    def __call__(self,
                 verts_padded,
                 faces_packed,
                 gt_point_clouds,
                 gt_point_clouds_lengths,
                 min_index,
                 max_index,
                 **kwargs):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
        assert len(verts_padded) == len(gt_point_clouds) == len(gt_point_clouds_lengths)
        sample_point_clouds = torch.zeros_like(gt_point_clouds)
        for i, verts_packed in enumerate(verts_padded):
            sample_point_cloud = self._sample_point_clouds(verts_packed, faces_packed, gt_point_clouds_lengths[i])
            sample_point_clouds[i, :sample_point_cloud.shape[0]] = sample_point_cloud

        chamfer_distances = self.scale * chamfer_distance(
            sample_point_clouds[min_index:max_index],
            gt_point_clouds[min_index:max_index],
            x_lengths=gt_point_clouds_lengths[min_index:max_index],
            y_lengths=gt_point_clouds_lengths[min_index:max_index],
            batch_reduction=None,
        )[0]
        self.chamfer_distances = chamfer_distances

        return chamfer_distances


class DepthMapRMSE:
    def __init__(self,
                 scale: float = 1.0,
                 inpaint_radius: int = 3,
                 with_boundary: bool = False,
                 **kwargs):
        self.scale = scale
        self.inpaint_radius = inpaint_radius
        self.with_boundary = with_boundary

    def _remove_outliers_fill_zeros(self, depth, mask, inpaint_radius=3):
        """
        use cv2.inpaint to fill the incorrect holes in gt_depth
        """
        depth_inpainted_list = []
        for i in range(depth.shape[0]):
            depth_np = depth[i].detach().cpu().numpy().astype(np.float32)
            mask_np = mask[i].detach().cpu().numpy().astype(np.uint8)
            holes = ((mask_np == 1) & (depth_np == 0)).astype(np.uint8)
            depth_inpainted = cv2.inpaint(depth_np, holes, inpaint_radius, cv2.INPAINT_TELEA)
            depth_inpainted_torch = torch.tensor(depth_inpainted, device=depth.device)
            depth_inpainted_list.append(depth_inpainted_torch)
        depth_inpainted_torch = torch.stack(depth_inpainted_list, dim=0)[..., None]
        return depth_inpainted_torch

    def __call__(self,
                 render_depth,
                 render_mask,
                 gt_depth,
                 gt_mask,
                 **kwargs):
        gt_depth = self._remove_outliers_fill_zeros(gt_depth, gt_mask, inpaint_radius=self.inpaint_radius)

        if self.with_boundary:
            gt_depth = gt_depth * gt_mask
            render_depth = render_depth * render_mask
        else:
            mask_intersection = gt_mask * render_mask
            gt_depth = gt_depth * mask_intersection
            render_depth = render_depth * mask_intersection

        gt_depth[gt_depth < 0] = 0
        render_depth[render_depth < 0] = 0
        errors = vmap(lambda x, y: torch.sqrt(torch.mean((x - y) ** 2)))(render_depth, gt_depth)
        errors = self.scale * errors
        self.errors = errors

        return errors
