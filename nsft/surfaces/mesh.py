from .base_surface import BaseSurface
import torch
import numpy as np
import igl
from pathlib import Path
from typing import Union
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
import imageio.v3 as iio
from plyfile import PlyData


def _unbind(z, n=2):
    out = np.split(z, n, axis=-1)
    for i in range(n):
        out[i] = out[i].squeeze()
    return out


class Mesh:
    _int_attrs = [
        "faces_packed",
        "E",
        "EMAP",
        "EF",
        "EI",
    ]
    _attrs = [
        "verts_packed",
        "edge_lengths_packed",
        "eig_vals",
        "adj_mat",
        "faces_areas_packed",
    ]
    def __init__(self,
        verts_packed: np.array,
        faces_packed: np.array,
        backend: str = "pytorch",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.verts_packed = verts_packed
        self.faces_packed = faces_packed
        # dihedral angle info
        self.E, self.EMAP, self.EF, self.EI = igl.edge_flaps(faces_packed)
        # edge length info
        self.edge_lengths_packed = np.linalg.norm(verts_packed[self.E[:, 0]] - verts_packed[self.E[:, 1]], axis=-1)
        # mesh_inextensibility info
        self.adj_mat = self._get_adjacency_matrix(self.E)
        self.eig_vals = self._get_eig_vals(verts_packed, self.adj_mat)
        # area info
        self.faces_areas_packed = self._get_faces_areas_padded(
            verts_padded=verts_packed[None],
            faces_packed=faces_packed
        ).squeeze(0)

        # parse backend
        if backend.lower() == "pytorch":
            self.to_pytorch(device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Backend {backend} not implemented.")

    def get_edge_lengths(self, verts_padded, edges_packed):
        return (verts_padded[..., edges_packed[:, 0], :] - verts_padded[..., edges_packed[:, 1], :]).norm(dim=-1)

    def get_faces_areas_and_normals_padded(self, verts_padded, faces_packed):
        i, j, k = faces_packed.unbind(dim=-1)
        Pi, Pj, Pk = verts_padded[..., i, :], verts_padded[..., j, :], verts_padded[..., k, :]
        normals = torch.cross(Pk - Pj, Pi - Pk, dim=-1)
        dbl_area = torch.linalg.norm(normals, dim=-1)
        return dbl_area / 2., normals / dbl_area[..., None]

    def get_faces_areas_padded(self, verts_padded, faces_packed):
        faces_areas_padded, _normals_padded = self.get_faces_areas_and_normals_padded(verts_padded, faces_packed)
        return faces_areas_padded

    def _get_faces_areas_and_normals_padded(self, verts_padded, faces_packed):
        i, j, k = _unbind(faces_packed, n=3)
        Pi, Pj, Pk = verts_padded[..., i, :], verts_padded[..., j, :], verts_padded[..., k, :]
        normals = np.cross(Pk - Pj, Pi - Pk, axis=-1)
        dbl_area = np.linalg.norm(normals, axis=-1)
        return dbl_area / 2., normals / dbl_area[..., None]

    def _get_faces_areas_padded(self, verts_padded, faces_packed):
        faces_areas_padded, _normals_padded = self._get_faces_areas_and_normals_padded(verts_padded, faces_packed)
        return faces_areas_padded

    def _get_adjacency_matrix(self, edges_packed: np.array):
        matrix = np.zeros((edges_packed.max() + 1, edges_packed.max() + 1))
        matrix[edges_packed[:, 0], edges_packed[:, 1]] = 1
        matrix[edges_packed[:, 1], edges_packed[:, 0]] = 1

        return matrix

    def _get_eig_vals(self, verts_packed: np.array, adj_mat: np.array):
        deg = adj_mat.sum(axis=-1)
        deg = np.where(deg == 0, 1, deg)

        mean_neighbors = (adj_mat @ verts_packed) / deg[..., None]
        diff = verts_packed[None] - mean_neighbors[:, None]
        diff = diff * adj_mat[..., None]
        cov = np.matmul(np.transpose(diff, axes=(0, 2, 1)), diff) / deg[..., None, None]
        eig_vals = np.linalg.eigvalsh(cov)
        return eig_vals

    def to_pytorch(self, device: Union[str, torch.device], dtype: torch.dtype):
        for attr in self._attrs:
            setattr(
                self,
                attr,
                torch.tensor(getattr(self, attr), device=device, dtype=dtype)
            )
        for attr in self._int_attrs:
            setattr(
                self,
                attr,
                torch.tensor(getattr(self, attr), device=device, dtype=torch.int32)
            )


class Pytorch3DTexturedMesh(Mesh, BaseSurface, name="Pytorch3DTexturedMesh"):
    def __init__(self,
        surface_path: Union[str, Path],
        texture_path: Union[Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        BaseSurface.__init__(self)
        meshes, texture = self.read_mesh(surface_path, texture_path, device=device, dtype=dtype)
        Mesh.__init__(
            self,
            verts_packed=meshes.verts_packed().detach().cpu().numpy(),
            faces_packed=meshes.faces_packed().detach().cpu().numpy(),
            device=device,
        )
        self.verts_uvs_packed = meshes.textures._verts_uvs_padded.squeeze(0).to(device=device, dtype=dtype)
        self.faces_uvs_packed = meshes.textures._faces_uvs_padded.squeeze(0).to(device=device, dtype=torch.int32)
        self.texture = texture

    def read_mesh(self, obj_path: Union[Path, str], texture_path: Union[Path], device: str, dtype: torch.dtype):
        verts, faces, aux = load_obj(obj_path, device=device)
        verts_uvs_padded = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs_padded = faces.textures_idx[None, ...]  # (1, F, 3)

        if verts_uvs_padded.max() > 1:
            verts_uvs_padded = verts_uvs_padded / verts_uvs_padded.max()

        texture = iio.imread(texture_path)  # range: [0, 255]
        texture = torch.from_numpy(texture).to(dtype=dtype, device=device) / 255.
        tex = TexturesUV(verts_uvs=verts_uvs_padded, faces_uvs=faces_uvs_padded, maps=texture[None, ...])

        meshes = Meshes(verts=verts[None], faces=faces.verts_idx[None], textures=tex)
        return meshes, texture


# class Pytorch3DUntexturedMesh(Mesh, BaseSurface, name="Pytorch3DUntexturedMesh"):
#     def __init__(self,
#         surface_path: Union[str, Path],
#         texture_path: Union[Path],
#         device: str = "cpu",
#         dtype: torch.dtype = torch.float32,
#         R: torch.tensor = None,
#         T: torch.tensor = None,
#         K: torch.tensor = None,
#         **kwargs
#     ):
#         BaseSurface.__init__(self)
#         verts_packed, faces_packed, texture = self.read_mesh(surface_path, texture_path, device=device, dtype=dtype)
#         Mesh.__init__(
#             self,
#             verts_packed=verts_packed.detach().cpu().numpy(),
#             faces_packed=faces_packed.detach().cpu().numpy(),
#             device=device,
#         )
#         if R is None:
#             R = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], device=device)
#         if T is None:
#             T = torch.zeros(3, device=device)
#         if K is None:
#             raise ValueError("Camera intrinsics K is required for untextured mesh.")
#         self.verts_uvs_packed, self.texture = generate_texture_map(
#             verts_packed=verts_packed,
#             texture=texture,
#             R=R, T=T, K=K,
#             device=device,
#             dtype=dtype
#         )
#         self.faces_uvs_packed = self.faces_packed
#
#
#     def read_mesh(self, obj_path: Union[Path, str], texture_path: Union[Path, str], device: str, dtype: torch.dtype):
#         verts, faces, aux = load_obj(obj_path, device=device)
#         verts.to(dtype=dtype)
#
#         texture = iio.imread(texture_path)  # range: [0, 255]
#         texture = torch.from_numpy(texture).to(dtype=dtype, device=device) / 255.
#
#         return verts, faces.verts_idx, texture


class TexturedMesh(Mesh, BaseSurface, name="TexturedMesh"):
    def __init__(self,
        surface_path: Union[str, Path],
        texture_path: Union[Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        BaseSurface.__init__(self)
        meshes, texture = self.read_mesh(surface_path, texture_path, device=device, dtype=dtype)
        Mesh.__init__(
            self,
            verts_packed=meshes.verts_packed().detach().cpu().numpy(),
            faces_packed=meshes.faces_packed().detach().cpu().numpy(),
            device=device,
        )
        self.verts_uvs_packed = meshes.textures._verts_uvs_padded.squeeze(0).to(device=device, dtype=dtype)
        self.faces_uvs_packed = meshes.textures._faces_uvs_padded.squeeze(0).to(device=device, dtype=torch.int32)
        self.texture = texture

    def read_mesh(self, surface_path: Union[Path, str], texture_path: Union[Path], device: str, dtype: torch.dtype):
        if isinstance(surface_path, str):
            surface_path = Path(surface_path)

        texture = iio.imread(texture_path)  # range: [0, 255]
        texture = torch.from_numpy(texture).to(dtype=dtype, device=device) / 255.

        if surface_path.suffix.endswith(".obj"):
            verts, faces, aux = load_obj(surface_path, device=device)
            verts_packed = verts
            faces_packed = faces.verts_idx
            verts_uvs_padded = aux.verts_uvs[None, ...]  # (1, V, 2)
            if verts_uvs_padded.max() > 1:
                verts_uvs_padded = verts_uvs_padded / verts_uvs_padded.max()
            faces_uvs_padded = faces.textures_idx[None, ...]  # (1, F, 3)

            tex = TexturesUV(verts_uvs=verts_uvs_padded, faces_uvs=faces_uvs_padded, maps=texture[None, ...])
        elif surface_path.suffix.endswith(".ply"):
            verts_packed, faces_packed, verts_uvs_packed, faces_uvs_packed = list(map(
                lambda x: torch.tensor(x, device=device),
                load_ply(surface_path)
            ))
            if verts_uvs_packed.max() > 1:
                verts_uvs_packed = verts_uvs_packed / verts_uvs_packed.max()

            tex = TexturesUV(verts_uvs=verts_uvs_packed[None], faces_uvs=faces_uvs_packed[None], maps=texture[None, ...])
        else:
            raise NotImplementedError(f"File format {surface_path.suffix} not implemented.")
        meshes = Meshes(verts=verts_packed[None], faces=faces_packed[None], textures=tex)
        return meshes, texture



class UntexturedMesh(Mesh, BaseSurface, name="UntexturedMesh"):
    def __init__(self,
        surface_path: Union[str, Path] = None,
        vertex_path: Union[str, Path] = None,
        face_path: Union[str, Path] = None,
        texture_path: Union[str, Path] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        R: torch.tensor = None,
        T: torch.tensor = None,
        K: torch.tensor = None,
        kinect: bool = False,
        **kwargs
    ):
        BaseSurface.__init__(self)
        verts_packed, faces_packed, texture = self.read_mesh(surface_path, vertex_path, face_path, texture_path)
        texture = torch.from_numpy(texture).to(dtype=dtype, device=device)

        Mesh.__init__(
            self,
            verts_packed=verts_packed,
            faces_packed=faces_packed,
            device=device,
        )
        if R is None:
            R = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
        if T is None:
            T = torch.zeros(3, device=device)
        if K is None:
            raise ValueError("Camera intrinsics K is required for untextured mesh.")
        self.verts_uvs_packed, self.texture = generate_texture_map(
            verts_packed=self.verts_packed.clone(),
            texture=texture,
            R=R, T=T, K=K,
            device=device, dtype=dtype,
            **kwargs
        )
        self.faces_uvs_packed = self.faces_packed

    def read_mesh(self, surface_path: Union[Path, str], vertex_path: Union[Path, str], face_path: Union[Path, str], texture_path: Union[Path, str]):
        if isinstance(surface_path, str):
            surface_path = Path(surface_path)
        if isinstance(vertex_path, str):
            vertex_path = Path(vertex_path)
        if isinstance(face_path, str):
            face_path = Path(face_path)

        if surface_path is not None:
            verts, faces, aux = load_obj(surface_path)
            verts_packed = verts.detach().cpu().numpy()
            faces_packed = faces.verts_idx.detach().cpu().numpy()
        else:
            if vertex_path.suffix.endswith(".pts"):
                verts_packed = np.loadtxt(vertex_path)
            else:
                raise NotImplementedError(f"File format {vertex_path.suffix} not implemented.")
            if face_path.suffix.endswith(".tri"):
                faces_packed = np.loadtxt(face_path, dtype=np.int32)
            else:
                raise NotImplementedError(f"File format {face_path.suffix} not implemented.")

        texture = iio.imread(texture_path) / 255.

        return verts_packed, faces_packed, texture


def generate_texture_map(
    verts_packed: torch.tensor,
    texture: torch.tensor,
    R: torch.tensor,
    T: torch.tensor,
    K: torch.tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    kinect: bool = False,
    cameras = None,
    **kwargs
):
    height, width, _3 = texture.shape

    # Project mesh vertices to generate UV coordinates
    verts_packed = verts_packed.clone()

    verts_cam_packed = verts_packed @ R.T + T
    verts_uvs_packed = verts_cam_packed @ K.T
    verts_uvs_packed = verts_uvs_packed[..., :2] / verts_uvs_packed[..., 2:]
    verts_uvs_packed[..., 0] = verts_uvs_packed[..., 0] / width
    verts_uvs_packed[..., 1] = 1. - verts_uvs_packed[..., 1] / height

    # verts_packed[..., (0, 1)] *= -1.
    # verts_uvs_packed = cameras.transform_points_screen(verts_packed, image_size=(height, width))[..., :2]
    # verts_uvs_packed = torch.div(verts_uvs_packed, torch.tensor([width, height], device=device, dtype=dtype), out=verts_uvs_packed)
    # texture = torch.flip(texture, dims=[0])

    return verts_uvs_packed, texture

def load_ply(filename):
    plydata = PlyData.read(filename)
    verts_packed = np.array([list(vertex) for vertex in plydata['vertex']])
    faces_packed = np.array([list(face[0]) for face in plydata['face']])

    # Extract face-level UV coordinates
    verts_uvs_packed = None
    faces_uvs_packed = None
    if 'texcoord' in plydata['face'].data.dtype.names:
        # Flatten the texcoord list and interpret as UVs
        uvs = []
        faces_uv = []

        for face in plydata['face']:
            texcoords = face['texcoord']  # Raw texcoord data for this face
            uv_indices = []

            for i in range(0, len(texcoords), 2):
                uv = (texcoords[i], texcoords[i + 1])  # Extract (u, v) pairs
                if uv not in uvs:
                    uvs.append(uv)
                uv_indices.append(uvs.index(uv))

            faces_uv.append(uv_indices)

        verts_uvs_packed = np.array(uvs, dtype=np.float32)
        faces_uvs_packed = np.array(faces_uv, dtype=np.int32)
    return verts_packed, faces_packed, verts_uvs_packed, faces_uvs_packed