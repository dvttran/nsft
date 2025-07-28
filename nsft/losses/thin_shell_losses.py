# Ref: https://gitlab.com/numod/shell-energy
import torch


class ThinShellLoss:
    def __init__(self,
                 mu: float = 1.0,
                 lam: float = 1.0,
                 membrane_weight: float = 1.0,
                 bending_weight: float = 0.001,
                 **kwargs):
        self.mu = mu
        self.lam = lam
        self.membrane_weight = membrane_weight
        self.bending_weight = bending_weight

    def __call__(self,
                 verts_padded,
                 verts_packed_0,
                 faces_packed,
                 E,
                 EMAP,
                 EF,
                 EI,
                 **kwargs
                 ):
        membrane_energy = self.membrane_energy(
            verts_padded=verts_padded,
            verts_packed_0=verts_packed_0,
            faces_packed=faces_packed,
            mu=self.mu,
            lam=self.lam,
            **kwargs
        )
        bending_energy = self.bending_energy(
            verts_padded=verts_padded,
            verts_packed_0=verts_packed_0,
            faces_packed=faces_packed,
            E=E,
            EMAP=EMAP,
            EF=EF,
            EI=EI,
            **kwargs
        )
        return self.membrane_weight * membrane_energy + self.bending_weight * bending_energy

    def membrane_energy(self,
                 verts_padded,
                 verts_packed_0,
                 faces_packed,
                 mu: float = 1.0,
                 lam: float = 1.0,
                 **kwargs
                 ):

        # deformed vertices and edges
        temp = verts_padded[..., faces_packed[..., 0], :]  # (batch_size, num_faces, 3)
        ej = verts_padded[..., faces_packed[..., 1], :]
        ek = verts_padded[..., faces_packed[..., 2], :]

        ei = ek - ej
        ej = temp - ek
        ek = ei + ej

        li_squared = ei.norm(dim=-1).square()  # (batch_size, num_faces)
        lj_squared = ej.norm(dim=-1).square()
        lk_squared = ek.norm(dim=-1).square()

        area_squared = torch.cross(ei, ej, dim=-1).norm(dim=-1).square() / 4.  # (batch_size, num_faces)

        # undeformed vertices and edges
        temp = verts_packed_0[..., faces_packed[..., 0], :]  # (num_faces, 3)
        ej = verts_packed_0[..., faces_packed[..., 1], :]
        ek = verts_packed_0[..., faces_packed[..., 2], :]

        ei = ek - ej  # (num_faces, 3)
        ej = temp - ek
        ek = ei + ej

        area_squared_0 = torch.cross(ei, ej, dim=-1).norm(dim=-1).square() / 4.  # (num_faces, )
        area_0 = torch.sqrt(area_squared_0)  # (num_faces, )

        # trace term
        trace = (
            (ej * ek).sum(dim=-1) * li_squared
            + (ek * ei).sum(dim=-1) * lj_squared
            - (ei * ej).sum(dim=-1) * lk_squared
        )  # (batch_size, num_faces)

        return torch.sum(
                (mu / 8. * trace + lam / 4. * area_squared) / area_0
                - ((mu / 2. + lam / 4.) * torch.log(area_squared / area_squared_0) + mu + lam / 4.) * area_0,
                dim=-1
        ).mean()  # (batch_size,).mean()

    def bending_energy(self,
                 verts_padded,
                 verts_packed_0,
                 faces_packed,
                 E,
                 EMAP,
                 EF,
                 EI,
                 **kwargs):
        # check for boundary
        boundary = torch.logical_or(EF[..., 0] == -1, EF[..., 1] == -1)
        E = E[~boundary]
        EF = EF[~boundary]
        EI = EI[~boundary]

        pi = E[..., 0]
        pj = E[..., 1]
        pk = faces_packed[EF[..., 0], EI[..., 0]]
        pl = faces_packed[EF[..., 1], EI[..., 1]]

        # deformed geometry
        Pi = verts_padded[..., pi, :]
        Pj = verts_padded[..., pj, :]
        Pk = verts_padded[..., pk, :]
        Pl = verts_padded[..., pl, :]
        theta = self._dihedral_angle(Pi, Pj, Pk, Pl)

        # undeformed geometry
        Pi = verts_packed_0[..., pi, :]
        Pj = verts_packed_0[..., pj, :]
        Pk = verts_packed_0[..., pk, :]
        Pl = verts_packed_0[..., pl, :]
        theta_0 = self._dihedral_angle(Pi, Pj, Pk, Pl)

        # compute volume, length of edge and theta difference
        volume = (
                torch.linalg.cross(Pk - Pj, Pi - Pk).norm(dim=-1)
                + torch.linalg.cross(Pl - Pi, Pj - Pl).norm(dim=-1)
        ) / 2.
        edge_length_squared = ((Pj - Pi) ** 2).sum(dim=-1)
        del_theta = theta - theta_0

        return torch.sum(
            3. * del_theta * del_theta * edge_length_squared / volume,
            dim=-1
        ).mean()

    def _dihedral_angle(self, Pi, Pj, Pk, Pl):
        # compute dihedral angle
        nk = torch.cross(Pk - Pj, Pi - Pk, dim=-1)
        nk = nk / nk.norm(dim=-1, keepdim=True)

        nl = torch.cross(Pl - Pi, Pj - Pl, dim=-1)
        nl = nl / nl.norm(dim=-1, keepdim=True)

        cross_product = torch.cross(nk, nl, dim=-1)
        dot_product = (nk * nl).sum(dim=-1)
        sharing_edge = Pj - Pi
        sharing_edge = sharing_edge / sharing_edge.norm(dim=-1, keepdim=True)

        return torch.pi - torch.atan2((sharing_edge * cross_product).sum(dim=-1), dot_product)
