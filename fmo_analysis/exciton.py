import numpy as np
from typing import List, Tuple
from .util import Pigment, Config


def delete_pigment_ham(ham: np.ndarray, delete: int) -> np.ndarray:
    """Remove the pigment from the Hamiltonian (set row and column to zero)."""
    new_ham = ham.copy()
    new_ham[delete, :] = 0
    new_ham[:, delete] = 0
    return new_ham


def delete_pigment_pigs(pigs: List[Pigment], delete: int) -> List[Pigment]:
    """Remove the pigment from the list of pigments (set mu = 0)."""
    new_pigs = [p for p in pigs]
    p = new_pigs[delete]
    p.mu *= 0
    new_pigs[delete] = p
    return new_pigs


def delete_pigment(config: Config, ham: np.ndarray, pigs: List[Pigment]) -> Tuple[np.ndarray, List[Pigment]]:
    """Returns the Hamiltonian and pigments with the pigment deleted."""
    if config.delete_pig > 0:
        ham = delete_pigment_ham(ham, config.delete_pig)
        pigs = delete_pigment_pigs(pigs, config.delete_pig)
    return ham, pigs


def make_stick_spectra(config: Config, ham: np.ndarray, pigs: List[Pigment]):
    """Computes the stick spectra and eigenvalues/eigenvectors for the system."""
    ham, pigs = delete_pigment(config, ham, pigs)
    n_pigs = ham.shape[0]
    e_vals, e_vecs = np.linalg.eig(ham)
    pig_mus = np.zeros((n_pigs, 3))
    for i, p in enumerate(pigs):
        pig_mus[i, :] = pigs[i].mu
    exciton_mus = np.zeros_like(pig_mus)
    stick_abs = np.zeros(n_pigs)
    stick_cd = np.zeros(n_pigs)
    for i in range(n_pigs):
        exciton_mus[i, :] = np.sum(np.repeat(e_vecs[:, 0], 3).reshape((n_pigs, 3)) * pig_mus, axis=0)
        stick_abs[i] = np.dot(exciton_mus[i], exciton_mus[i])
        for j in range(n_pigs):
            for k in range(n_pigs):
                r = pigs[j].pos - pigs[k].pos
                mu_cross = np.cross(pigs[j].mu, pigs[k].mu)
                stick_cd[i] += e_vecs[j, i] * e_vecs[k, i] * np.dot(r, mu_cross)
    out = {
        "ham_deleted": ham,
        "pigs_deleted": pigs,
        "e_vals": e_vals,
        "e_vecs": e_vecs,
        "exciton_mus": exciton_mus,
        "stick_abs": stick_abs,
        "stick_cd": stick_cd
    }
    return out
    
    