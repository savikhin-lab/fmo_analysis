from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.linalg import lapack

from .util import Config, Pigment, faster_np_savetxt, faster_np_savetxt_1d
import ham2spec


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


def pigs_to_arrays(pigs: List[Pigment]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of pigments into separate arrays for positions and dipole moments."""
    n_pigs = len(pigs)
    mus = np.empty((n_pigs, 3))
    rs = np.empty((n_pigs, 3))
    for i, p in enumerate(pigs):
        mus[i] = p.mu
        rs[i] = p.pos
    return mus, rs


def confs_to_arrays(confs: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of confs into separate arrays for Hamiltonians, positions, and dipole moments."""
    n_confs = len(confs)
    n_pigs = confs[0]["ham"].shape[0]
    hams = np.empty((n_confs, n_pigs, n_pigs))
    mus = np.empty((n_confs, n_pigs, 3))
    rs = np.empty((n_confs, n_pigs, 3))
    for i, c in enumerate(confs):
        hams[i] = c["ham"]
        m, r = pigs_to_arrays(c["pigs"])
        mus[i] = m
        rs[i] = r
    return hams, mus, rs


def stick_spectrum(config, ham, pigs):
    """Computes the stick spectra and eigenvalues/eigenvectors for the system."""
    mus, rs = pigs_to_arrays(pigs)
    if config.normalize:
        total_dpm = np.sum([np.dot(mus[i], mus[i]) for i in range(len(pigs))])
        mus /= total_dpm
    if config.delete_pig > 0:
        idx = config.delete_pig - 1
        ham[:, idx] = 0
        ham[idx, :] = 0
        mus[idx, :] = 0
        rs[idx, :] = 0
    sticks = ham2spec.compute_stick_spectrum(ham, mus, rs)
    return sticks


def stick_spectra(config: Config, confs: List[Dict]) -> List[Dict]:
    """Computes the stick spectra for a collection of Hamiltonians"""
    hams, mus, rs = confs_to_arrays(confs)
    if config.delete_pig > 0:
        pig_index = config.delete_pig - 1
        hams[:, pig_index, :] = 0
        hams[:, :, pig_index] = 0
        mus[:, pig_index, :] = 0
        rs[:, pig_index, :] = 0
    sticks = ham2spec.compute_stick_spectra(hams, mus, rs)
    for s, c in zip(sticks, confs):
        try:
            # If there's a record of which file this conf came from,
            # pass it along.
            s["file"] = c["file"]
        except KeyError:
            # If there's no record, not a big deal
            pass
    return sticks


def save_stick_spectrum(outdir: Path, stick: Dict):
    """Saves the result of computing a stick spectrum to disk.
    
    This saves 5 files:
    - 'energies.csv'
    - 'eigenvectors.csv'
    - 'exciton_mus.csv'
    - 'stick_abs.csv'
    - 'stick_cd.csv'

    Note: The eigenvectors are stored one per column, but the exciton dipole moments
          are stored one per row.
    """
    # dir_name = stick["file"].stem
    # outdir = parent_dir / dir_name
    # outdir.mkdir(exist_ok=True)
    faster_np_savetxt_1d(outdir / "energies.csv", stick["e_vals"])
    faster_np_savetxt(outdir / "eigenvectors.csv", stick["e_vecs"])
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
    sticks_dir = outdir / "stick_spectra"
    sticks_dir.mkdir(exist_ok=True)
    for s in sticks:
        stick_dir = sticks_dir / s["file"].stem
        stick_dir.mkdir(exist_ok=True)
        save_stick_spectrum(stick_dir, s)


def broadened_spectrum_from_conf(config: Config, conf: Dict) -> Dict:
    """Make the broadened spectrum from a Hamiltonian."""
    ham = conf["ham"]
    mus, rs = pigs_to_arrays(conf["pigs"])
    if config.delete_pig > 0:
        idx = config.delete_pig - 1
        ham[:, idx] = 0
        ham[idx, :] = 0
        mus[idx, :] = 0
        rs[idx, :] = 0
    return ham2spec.compute_broadened_spectrum_from_ham(ham, mus, rs, config)


def het_broadened_spectrum_from_conf(config: Config, conf: Dict) -> Dict:
    """Make the broadened spectrum from a Hamiltonian with different bandwidths for each transition."""
    ham = conf["ham"]
    mus, rs = pigs_to_arrays(conf["pigs"])
    if config.delete_pig > 0:
        idx = config.delete_pig - 1
        ham[:, idx] = 0
        ham[idx, :] = 0
        mus[idx, :] = 0
        rs[idx, :] = 0
    return ham2spec.compute_het_broadened_spectrum_from_ham(ham, mus, rs, config)


def broadened_spectrum_from_stick(config: Config, stick: Dict) -> Dict:
    """Make the broadened spectrum from a stick spectrum."""
    return ham2spec.compute_broadened_spectrum_from_stick(stick, config)


def het_broadened_spectrum_from_stick(config: Config, stick: Dict) -> Dict:
    """Make the broadened spectrum from a stick spectrum with different bandwidths for each transition."""
    return ham2spec.compute_het_broadened_spectrum_from_stick(stick, config)


def broadened_spectrum_from_confs(config: Config, confs: List[Dict]) -> Dict:
    """Make the average broadened spectra from a collection on Hamiltonians."""
    hams, mus, rs = confs_to_arrays(confs)
    if config.delete_pig > 0:
        pig_index = config.delete_pig - 1
        hams[:, pig_index, :] = 0
        hams[:, :, pig_index] = 0
        mus[:, pig_index, :] = 0
        rs[:, pig_index, :] = 0
    return ham2spec.compute_broadened_spectrum_from_hams(hams, mus, rs, config)


def het_broadened_spectrum_from_confs(config: Config, confs: List[Dict]) -> Dict:
    """Make the average broadened spectra from a collection on Hamiltonians."""
    hams, mus, rs = confs_to_arrays(confs)
    if config.delete_pig > 0:
        pig_index = config.delete_pig - 1
        hams[:, pig_index, :] = 0
        hams[:, :, pig_index] = 0
        mus[:, pig_index, :] = 0
        rs[:, pig_index, :] = 0
    return ham2spec.compute_het_broadened_spectrum_from_hams(hams, mus, rs, config)


def average_broadened_spectra(specs: List[Dict]):
    n_specs = len(specs)
    x = specs[0]["x"]
    n_points = len(x)
    abs_data = np.empty((n_points, n_specs))
    cd_data = np.empty((n_points, n_specs))
    for i, s in enumerate(specs):
        abs_data[:, i] = s["abs"]
        cd_data[:, i] = s["cd"]
    avg = {
        "x": x,
        "abs": abs_data.mean(axis=1),
        "cd": cd_data.mean(axis=1),
    }
    return avg


def save_broadened_spectrum(outdir: Path, spec: Dict) -> None:
    """Save a single broadened spectrum at the top level of the output directory."""
    x = spec["x"]
    abs_data = np.zeros((len(x), 2))
    abs_data[:, 0] = x
    abs_data[:, 1] = spec["abs"]
    faster_np_savetxt(outdir / "abs.csv", abs_data)
    cd_data = np.zeros((len(x), 2))
    cd_data[:, 0] = x
    cd_data[:, 1] = spec["cd"]
    faster_np_savetxt(outdir / "cd.csv", cd_data)
    save_stacked_plot(outdir / "spec_wn.tiff", x, abs_data[:, 1], cd_data[:, 1])
    save_stacked_plot(outdir / "spec_nm.tiff", wavenumber_to_wavelength(x), abs_data[:, 1], cd_data[:, 1], xlabel="Wavelength (nm)")


def save_broadened_spectra(config: Config, outdir: Path, specs: List[Dict]) -> None:
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
    for s in specs:
        save_broadened_spectrum_csv(abs_dir, cd_dir, s)
    if config.save_figs:
        save_stacked_plots(plots_dir, specs)


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
