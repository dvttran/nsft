import random
import numpy as np
import torch
from torch.func import vmap
from loguru import logger


class MeshInexentsibilityLoss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, verts_padded, eig_vals, adj_mat, start_dim=1, **kwargs):
        cov = self.get_cov(verts_padded, adj_mat).unsqueeze(2).expand(-1, -1, 3, -1, -1)
        lambda_I = eig_vals.view(1, -1, 3, 1, 1) * torch.eye(3, device=cov.device)
        # return torch.linalg.det(cov - lambda_I).abs().flatten(start_dim=start_dim).sum(dim=-1)
        # return torch.linalg.det(cov - lambda_I).abs().sum(dim=-1).mean(dim=-1)
        return torch.linalg.det(cov - lambda_I).abs().sum(dim=-1).mean(dim=-1).mean()

    def get_cov(self, verts_padded, adj_mat, **kwargs):
        return vmap(lambda verts_packed: self._get_cov(verts_packed, adj_mat))(verts_padded)

    def _get_cov(self, verts_packed, adj_mat, **kwargs):
        deg = adj_mat.sum(dim=-1)
        deg[deg == 0] = 1

        mean_neighbors = (adj_mat @ verts_packed) / deg[..., None]
        diff = verts_packed.unsqueeze(0) - mean_neighbors.unsqueeze(1)
        diff = diff * adj_mat.unsqueeze(-1)
        cov = torch.bmm(diff.transpose(1, 2), diff) / deg[..., None, None]
        return cov


class AdaptiveMeshInexentsibilityLoss:
    def __init__(self, edge_lengths_0, q: float = 0.5, **kwargs):
        self.adaptive_weight = 1. / (torch.quantile(edge_lengths_0, q=q) ** 6)
        logger.info(f"adaptive_weight: {self.adaptive_weight}")
        self.mesh_inextensibility_loss = MeshInexentsibilityLoss()

    def __call__(self, **kwargs):
        return self.adaptive_weight * self.mesh_inextensibility_loss(**kwargs)


class EdgePreservationLoss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, verts_padded, edges_packed, edge_lengths_0, **kwargs):
        edge_lengths = torch.norm(verts_padded[:, edges_packed[:, 0]] - verts_padded[:, edges_packed[:, 1]], dim=-1)
        relative_diff_edge_length = torch.abs(edge_lengths - edge_lengths_0) / edge_lengths_0
        return relative_diff_edge_length.mean()


class AreaPreservationLoss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, faces_areas, face_areas_0, **kwargs):
        relative_diff_area = torch.abs(faces_areas - face_areas_0) / face_areas_0
        return relative_diff_area.mean()
