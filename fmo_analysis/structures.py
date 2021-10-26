import numpy as np
from typing import List, Tuple, Dict
from .util import Pigment


def confs_to_arrs(confs: List[Tuple[np.ndarray, List[Pigment]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the list of tuples of arrays into separate arrays."""
    n_confs = len(confs)
    n_pigs = len(confs[0][1])
    hams = np.zeros((n_confs, n_pigs, n_pigs))
    coords = np.zeros((n_confs, n_pigs, 3))
    mus = np.zeros((n_confs, n_pigs, 3))
    for i, (h, ps) in enumerate(confs):
        hams[i, :, :] = h
        for j, p in enumerate(ps):
            coords[i, j, :] = p.pos
            mus[i, j, :] = p.mu
    return hams, coords, mus


def center_structures(coords: np.ndarray) -> np.ndarray:
    """Translate the centers of mass for each structure to the origin."""
    coms = coords.mean(axis=1)
    avg_com = coms.mean(axis=0)
    centered = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        centered[i, :, :] = coords[i, :, :] - coms[i]
    return centered


def find_rotation_matrix(this_structure: np.ndarray, target_structure: np.ndarray) -> np.ndarray:
    """Finds the 3x3 rotation matrix that transforms 'this_structure' into 'target_structure'.
    
    Notes:
    - Assumes that both structures have been centered on the origin.
    
    Implementation taken from:
    https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    https://nghiaho.com/?page_id=671
    """
    # '@' is matrix multiplication
    # Algorithm expects 3x8 arrays, but we have 8x3 so we transpose:
    # this_structure -> this_structure.T
    # target_structure.T -> target_structure
    H = this_structure.T @ target_structure
    U, S, V_trans = np.linalg.svd(H)
    R = V_trans.T @ U.T
    if np.linalg.det(R) < 0:
        V_trans[2, :] *= -1
        R = V_trans.T @ U.T
    return R


def rotate(coords: np.ndarray, mus: np.ndarray, max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the structures to best align with each other."""
    n_structures, _, _ = coords.shape
    avg_structure = np.mean(coords, axis=0)  # average the location of each pigment as a starting point
    err_old = 1e50  # Initial error between current structure and the average
    for _ in range(max_iter):
        for i in range(n_structures):
            R = find_rotation_matrix(coords[i, :, :], avg_structure)
            coords[i, :, :] = coords[i, :, :] @ R.T  # '@' is the matrix multiplication operator
            mus[i, :, :] = mus[i, :, :] @ R.T
        err = np.std(coords - avg_structure)
        if abs(err_old - err) / err < tol:
            break
        avg_structure = np.mean(coords, axis=0)
        err_old = err
    return coords, mus


def plot_pig_positions(coords: np.ndarray, **opts: Dict):
    """Plots the locations of all pigments in 3D.
    
    This is really just here so you can visually inspect whether alignment has worked."""
    pass