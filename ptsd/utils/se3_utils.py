import torch
from typing import Tuple


def remove_mean(samples, n_particles, n_dimensions):
    """Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples


def interatomic_dist(samples):
    n_particles = samples.shape[-2]
    # Compute the pairwise differences and distances
    distances = samples[:, None, :, :] - samples[:, :, None, :]
    distances = distances[
        :, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1
    ]

    dist = torch.linalg.norm(distances, dim=-1)

    return dist


def find_alignment_kabsch(
    P: torch.Tensor, Q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find alignment using Kabsch algorithm between two sets of points P and Q.
    Args:
    P (torch.Tensor): A tensor of shape (N, 3) representing the first set of points.
    Q (torch.Tensor): A tensor of shape (N, 3) representing the second set of points.
    Returns:
    Tuple[Tensor, Tensor]: A tuple containing two tensors, where the first tensor is the rotation matrix R
    and the second tensor is the translation vector t. The rotation matrix R is a tensor of shape (3, 3)
    representing the optimal rotation between the two sets of points, and the translation vector t
    is a tensor of shape (3,) representing the optimal translation between the two sets of points.
    """
    # Shift points w.r.t centroid
    centroid_P, centroid_Q = P.mean(dim=0), Q.mean(dim=0)
    P_c, Q_c = P - centroid_P, Q - centroid_Q
    # Find rotation matrix by Kabsch algorithm
    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    # ensure right-handedness
    d = torch.sign(torch.linalg.det(V @ U.T))
    # Trick for torch.vmap
    if P.shape[-1] == 3:
        diag_values = torch.cat(
            [
                torch.ones(1, dtype=P.dtype, device=P.device),
                torch.ones(1, dtype=P.dtype, device=P.device),
                d * torch.ones(1, dtype=P.dtype, device=P.device),
            ]
        )
    elif P.shape[-1] == 2:
        diag_values = torch.cat(
            [
                torch.ones(1, dtype=P.dtype, device=P.device),
                d * torch.ones(1, dtype=P.dtype, device=P.device),
            ]
        )
    else:
        print("unsupport dim for kabsch")
        raise ValueError
    # This is only [[1,0,0],[0,1,0],[0,0,d]]
    M = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device) * diag_values
    R = V @ M @ U.T
    # Find translation vectors
    t = centroid_Q[None, :] - (R @ centroid_P[None, :].T).T
    t = t.T
    return R, t.squeeze()


def calculate_rmsd(pos: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref.
    Args:
    pos (torch.Tensor): A tensor of shape (N, 3) representing the positions of the first set of points.
    ref (torch.Tensor): A tensor of shape (N, 3) representing the positions of the second set of points.
    Returns:
    torch.Tensor: RMSD between the two sets of points.
    """
    if pos.shape[0] != ref.shape[0]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch(ref, pos)
    ref0 = (R @ ref.T).T + t
    rmsd = torch.linalg.norm(ref0 - pos, dim=1).mean()
    return rmsd


def calculate_rmsd_matrix(R_ref: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    fn_vmap_row = torch.vmap(calculate_rmsd, in_dims=(0, None))
    fn_vmap_row_col = torch.vmap(fn_vmap_row, in_dims=(None, 0))
    return fn_vmap_row_col(R, R_ref)


def avg_distance_to_origin(samples, n_particles, n_dimensions):
    # remove mean
    samples = remove_mean(samples, n_particles, n_dimensions)
    batch_shape = samples.shape[
        0
    ]  # here assuming that the first dimension is the batch size
    samples = samples.view(batch_shape, n_particles, n_dimensions)
    return torch.mean(torch.linalg.norm(samples, dim=-1), dim=1)
