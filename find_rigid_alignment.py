# from https://johnwlambert.github.io/icp/

from typing import Tuple
import numpy as np

def find_rigid_alignment(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.

    Args:
        A: Array of shape (N,D) -- Reference Point Cloud (target)
        B: Array of shape (N,D) -- Point Cloud to Align (source)

    Returns:
        R: optimal rotation (3,3)
        t: optimal translation (3,)
    """
    num_pts = A.shape[0]
    dim = A.shape[1]

    a_mean = np.mean(A, axis=0)
    b_mean = np.mean(B, axis=0)

    # Zero-center the point clouds
    A -= a_mean
    B -= b_mean

    N = np.zeros((dim, dim))
    for i in range(num_pts):
        N += A[i].reshape(dim,1) @ B[i].reshape(1,dim)
    N = A.T @ B

    U, D, V_T = np.linalg.svd(N)
    S = np.eye(dim)
    det = np.linalg.det(U) * np.linalg.det(V_T.T)

    # Check for reflection case
    if not np.isclose(det,1.):
        S[dim-1,dim-1] = -1

    R = U @ S @ V_T
    t = R @ b_mean.reshape(dim,1) - a_mean.reshape(dim,1)
    return R, -t.squeeze()
