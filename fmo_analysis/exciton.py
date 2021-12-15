from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.linalg import lapack

from .util import Config, Pigment, faster_np_savetxt, faster_np_savetxt_1d


def delete_pigment_ham(ham: np.ndarray, delete_pig: int) -> np.ndarray:
    """Remove the pigment from the Hamiltonian (set row and column to zero)."""
    new_ham = ham.copy()
    idx = delete_pig - 1
    new_ham[idx, :] = 0
    new_ham[:, idx] = 0
    return new_ham


def delete_pigment_pigs(pigs: List[Pigment], delete_pig: int) -> List[Pigment]:
    """Remove the pigment from the list of pigments (set mu = 0)."""
    idx = delete_pig - 1
    p = pigs[idx]
    p.mu *= 0
    pigs[idx] = p
    return pigs


def delete_pigment(config: Config, ham: np.ndarray, pigs: List[Pigment]) -> Tuple[np.ndarray, List[Pigment]]:
    """Returns the Hamiltonian and pigments with the pigment deleted."""
    if config.delete_pig > 0:
        ham = delete_pigment_ham(ham, config.delete_pig)
        pigs = delete_pigment_pigs(pigs, config.delete_pig)
    return ham, pigs


def make_stick_spectrum(config: Config, ham: np.ndarray, pigs: List[Pigment]) -> Dict:
    """Computes the stick spectra and eigenvalues/eigenvectors for the system."""
    ham, pigs = delete_pigment(config, ham, pigs)
    n_pigs = ham.shape[0]
    if config.delete_pig > n_pigs:
        raise ValueError(f"Tried to delete pigment {config.delete_pig} but system only has {n_pigs} pigments.")
    e_vals_fortran_order, _, _, e_vecs_fortran_order, _ = lapack.sgeev(ham)
    e_vals = np.ascontiguousarray(e_vals_fortran_order)
    e_vecs = np.ascontiguousarray(e_vecs_fortran_order)
    pig_mus = np.zeros((n_pigs, 3))
    if config.normalize:
        total_dpm = np.sum([np.dot(p.mu, p.mu) for p in pigs])
        for i in range(len(pigs)):
            pigs[i].mu /= total_dpm
    for i, p in enumerate(pigs):
        pig_mus[i, :] = pigs[i].mu
    exciton_mus = np.zeros_like(pig_mus)
    stick_abs = np.zeros(n_pigs)
    stick_cd = np.zeros(n_pigs)
    r_mu_cross_cache = make_r_dot_mu_cross_cache(pigs)
    for i in range(n_pigs):
        exciton_mus[i, :] = np.sum(np.repeat(e_vecs[:, i], 3).reshape((n_pigs, 3)) * pig_mus, axis=0)
        stick_abs[i] = np.dot(exciton_mus[i], exciton_mus[i])
        energy = e_vals[i]
        if energy == 0:
            # If the energy is zero, the pigment has been deleted
            energy = 100_000
        wavelength = 1e8 / energy  # in angstroms
        stick_coeff = 2 * np.pi / wavelength
        e_vec_weights = make_weight_matrix(e_vecs, i)
        stick_cd[i] = 2 * stick_coeff * np.dot(e_vec_weights.flatten(), r_mu_cross_cache.flatten())
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


def make_r_dot_mu_cross_cache(pigs):
    """Computes a cache of (r_i - r_j) * (mu_i x mu_j)"""
    n = len(pigs)
    cache = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            r_i = pigs[i].pos
            r_j = pigs[j].pos
            r_ij = r_i - r_j
            mu_i = pigs[i].mu
            mu_j = pigs[j].mu
            mu_ij_cross = np.empty(3)
            mu_ij_cross[0] = mu_i[1] * mu_j[2] - mu_i[2] * mu_j[1]
            mu_ij_cross[1] = mu_i[2] * mu_j[0] - mu_i[0] * mu_j[2]
            mu_ij_cross[2] = mu_i[0] * mu_j[1] - mu_i[1] * mu_j[0]
            cache[i, j] = r_ij[0] * mu_ij_cross[0] + r_ij[1] * mu_ij_cross[1] + r_ij[2] * mu_ij_cross[2]
    return cache


def make_weight_matrix(e_vecs, col):
    """Makes the matrix of weights for CD from the eigenvectors"""
    n = e_vecs.shape[0]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = e_vecs[i, col] * e_vecs[j, col]
    return mat


def make_stick_spectra(config: Config, cf: List[Dict]) -> List[Dict]:
    """Computes the OD and CD stick spectra for a single Hamiltonian and set of pigments."""
    results = []
    for c in cf:
        stick = make_stick_spectrum(config, c["ham"], c["pigs"])
        try:
            # If there's a record of which file this conf came from,
            # pass it along.
            stick["file"] = c["file"]
        except KeyError:
            # If there's no record, not a big deal
            pass
        results.append(stick)
    return results


def save_stick_spectrum(parent_dir: Path, stick: Dict):
    """Saves the result of computing a stick spectrum to disk.
    
    This saves 5 files:
    - 'energies.csv'
    - 'eigenvectors.csv'
    - 'exciton_mus.csv'
    - 'stick_abs.csv'
    - 'stick_cd.csv'

    The convention for the data stored in each file is that row X in each file
    corresponds to the same exciton. For example, the first entry in 'energies.csv'
    is the energy of the first exciton, and the first ROW in 'eigenvectors.csv' is
    the eigenvector of the first exciton. This means that the array in 'eigenvectors.csv'
    is stored in the transposed order from how it is generated by np.linalg.eig().
    """
    dir_name = stick["file"].stem
    outdir = parent_dir / dir_name
    outdir.mkdir(exist_ok=True)
    faster_np_savetxt_1d(outdir / "energies.csv", stick["e_vals"])
    faster_np_savetxt(outdir / "eigenvectors.csv", stick["e_vecs"].T)
    faster_np_savetxt(outdir / "exciton_mus.csv", stick["exciton_mus"])
    faster_np_savetxt_1d(outdir / "stick_abs.csv", stick["stick_abs"])
    faster_np_savetxt_1d(outdir / "stick_cd.csv", stick["stick_cd"])


def save_stick_spectra(outdir: Path, sticks: List[Dict]) -> None:
    """Saves results of stick spectra computation.
    
    The directory structure is:
    <output directory>/
        stick_spectra/
            conf*/
                <result>
    """
    stick_dir = outdir / "stick_spectra"
    stick_dir.mkdir(exist_ok=True)
    for s in sticks:
        save_stick_spectrum(stick_dir, s)


def make_broadened_spectrum(config: Config, stick: Dict) -> Dict:
    """Make the broadened spectrum from a stick spectrum."""
    x = np.arange(config.xfrom, config.xto, config.xstep, dtype=np.float64)
    abs = np.zeros_like(x)
    cd = np.zeros_like(x)
    dip_strengths = stick["stick_abs"]
    rot_strengths = stick["stick_cd"]
    energies = stick["e_vals"]
    sigma_squared = config.bandwidth**2 / (4 * np.log(2))
    for exc in range(len(energies)):
        abs += dip_strengths[exc] * np.exp(-(x - energies[exc])**2 / sigma_squared)
        cd += rot_strengths[exc] * np.exp(-(x - energies[exc])**2 / sigma_squared)
    return {"abs": abs, "cd": cd, "x": x}


def make_broadened_spectra(config: Config, sticks: List[Dict]) -> Dict:
    """Make broadened spectra from the stick spectra."""
    individual_spectra = []
    for s in sticks:
        b = make_broadened_spectrum(config, s)
        try:
            b["file"] = s["file"]
        except KeyError:
            # If the filename isn't present, not a big deal
            pass
        individual_spectra.append(b)
    avg_abs = np.mean([s["abs"] for s in individual_spectra], axis=0)
    avg_cd = np.mean([s["cd"] for s in individual_spectra], axis=0)
    return {"spectra": individual_spectra, "avg_abs": avg_abs, "avg_cd": avg_cd}


def save_broadened_spectra(config: Config, outdir: Path, b_specs: List[Dict]) -> None:
    """Save the results of computing the broadened spectra.
    
    The directory structure is:
    <output directory>/
        broadened_spectra/
            abs/
            cd/
            plots/
            avg_abs.csv
            avg_cd.csv
            avg.png
    """
    b_dir = outdir / "broadened_spectra"
    b_dir.mkdir(exist_ok=True)
    abs_dir = b_dir / "abs"
    abs_dir.mkdir(exist_ok=True)
    cd_dir = b_dir / "cd"
    cd_dir.mkdir(exist_ok=True)
    plots_dir = b_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    if config.save_intermediate:
        for s in b_specs["spectra"]:
            save_broadened_spectrum_csv(abs_dir, cd_dir, s)
    if config.save_figs:
        save_stacked_plots(plots_dir, b_specs["spectra"])
    x = b_specs["spectra"][0]["x"]
    abs_data = np.zeros((len(x), 2))
    abs_data[:, 0] = x
    abs_data[:, 1] = b_specs["avg_abs"]
    faster_np_savetxt(outdir / "abs.csv", abs_data)
    cd_data = np.zeros((len(x), 2))
    cd_data[:, 0] = x
    cd_data[:, 1] = b_specs["avg_cd"]
    faster_np_savetxt(outdir / "cd.csv", cd_data)
    save_stacked_plot(b_dir / "avg.tiff", x, abs_data[:, 1], cd_data[:, 1], title="Average")
    save_stacked_plot(b_dir / "avg_nm.tiff", wavenumber_to_wavelength(x), abs_data[:, 1], cd_data[:, 1], title="Average", xlabel="Wavelength (nm)")


def save_broadened_spectrum_csv(a_dir: Path, c_dir: Path, spec: Dict) -> None:
    """Save CSVs of the broadened absorption and CD spectra."""
    x = spec["x"]
    abs = spec["abs"]
    cd = spec["cd"]
    stem = spec["file"].stem
    faster_np_savetxt(a_dir / f"{stem}_abs.csv", np.stack((x, abs), axis=1))
    faster_np_savetxt(c_dir / f"{stem}_cd.csv", np.stack((x, cd), axis=1))


def save_stacked_plots(outdir: Path, specs: List[Dict], **opts: Dict) -> None:
    """Save stacked plots quickly by reusing axes and a single figure object."""
    x = specs[0]["x"]
    max_abs = max([np.max(s["abs"]) for s in specs])
    min_abs = min([np.min(s["abs"]) for s in specs])
    max_cd = max([np.max(s["cd"]) for s in specs])
    min_cd = min([np.min(s["cd"]) for s in specs])
    fig, (ax_abs, ax_cd) = plt.subplots(2, 1, sharex=True)
    ax_abs.set(xlabel="", ylabel="Abs.")
    try:
        ax_cd.set(xlabel=opts["xlabel"], ylabel="CD")
    except KeyError:
        ax_cd.set(xlabel="Wavenumbers (cm^-1)", ylabel="CD")
    try:
        ax_abs.set(title=opts["title"])
    except KeyError:
        pass
    # Prepare the axes beforehand
    line_abs = ax_abs.plot(x, specs[0]["abs"], animated=True)[0]
    line_cd = ax_cd.plot(x, specs[0]["cd"], animated=True)[0]
    ax_abs.set_xlim(x[0], x[-1])
    ax_abs.set_ylim(1.1 * min_abs, 1.1 * max_abs)
    ax_cd.set_ylim(1.1 * min_cd, 1.1 * max_cd)
    ax_abs.grid()
    ax_cd.grid()
    for s in specs:
        line_abs.set_ydata(s["abs"])
        line_cd.set_ydata(s["cd"])
        fig.canvas.draw()
        fig.canvas.flush_events()
        path = outdir / f'{s["file"].stem}.tiff'
        fig.savefig(path)
    plt.close(fig)


def save_stacked_plot(path: Path, x: np.ndarray, abs: np.ndarray, cd: np.ndarray, **opts: Dict) -> None:
    """Save a plot with absorption and CD in the same figure."""
    fig, (ax_abs, ax_cd) = plt.subplots(2, 1, sharex=True)
    ax_abs.plot(x, abs)
    ax_abs.set(xlabel="", ylabel="Abs.")
    ax_abs.grid()
    ax_cd.plot(x, cd)
    try:
        ax_cd.set(xlabel=opts["xlabel"], ylabel="CD")
    except KeyError:
        ax_cd.set(xlabel="Wavenumbers (cm^-1)", ylabel="CD")
    ax_cd.grid()
    try:
        ax_abs.set(title=opts["title"])
    except KeyError:
        pass
    fig.savefig(path)
    plt.close(fig)


def wavenumber_to_wavelength(wn: float) -> float:
    """Convert a wavenumber in cm^-1 to wavelength in nm."""
    return (1 / wn) * 1e7


def random_hams(ham: np.ndarray, std_devs: List[float], n_hams: int) -> np.ndarray:
    """Construct Hamiltonians where the diagonals are randomly shifted."""
    n_pigs, _ = ham.shape
    rng = default_rng(seed=123)
    rand_hams = np.tile(ham, (n_hams, 1, 1))
    for i in range(n_pigs):
        center = ham[i, i]
        std_dev = std_devs[i]
        diags = rng.normal(loc=center, scale=std_dev, size=n_hams)
        rand_hams[:, i, i] = diags
    return rand_hams


def apply_const_diag_shift(ham, shift):
    """Apply a constant shift along the diagonal of the Hamiltonian"""
    n, _ = ham.shape
    shifted = np.copy(ham)
    for i in range(n):
        shifted[i, i] += shift
    return shifted
