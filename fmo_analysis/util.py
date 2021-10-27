from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import List, Tuple


@dataclass(frozen=True)
class Config:
    xfrom: int
    xto: int
    xstep: int
    bandwidth: float
    shift_diag: float
    dip_cor: float
    delete_pig: int
    use_shift_T: bool
    scale: bool
    overwrite: bool
    save_figs: bool
    save_intermediate: bool
    empirical: bool


@dataclass
class Pigment:
    pos: np.ndarray
    mu: np.ndarray


def parse_conf_file(cf_path: Path) -> Tuple[np.ndarray, List[Pigment]]:
    """Extract the Hamiltonian and pigment data from a 'conf*.csv' file."""
    try:
        # Try loading with whitespace delimiting the columns
        arr = np.loadtxt(cf_path)
    except ValueError:
        # If that doesn't work then fall back to comma as the delimiter
        arr = np.loadtxt(cf_path, delimiter=",")
    n, _ = arr.shape
    ham = arr[:, :n]
    mus = arr[:, -6:-3]
    coords = arr[:, -3:]
    pigments = [Pigment(np.array(c), np.array(m)) for c, m in zip(coords, mus)]
    return ham, pigments


def save_conf_files(outdir: Path, filenames: List[str], hams: np.ndarray, coords: np.ndarray, mus: np.ndarray) -> None:
    """Save conf files from arrays."""
    _, n_pigs, _ = hams.shape
    for i, fname in enumerate(filenames):
        out_arr = np.zeros((n_pigs, n_pigs + 6))
        out_arr[:, :n_pigs] = hams[i, :, :]
        out_arr[:, -6:-3] = mus[i, :, :]
        out_arr[:, -3:] = coords[i, :, :]
        faster_np_savetxt(outdir / fname, out_arr)


def faster_np_savetxt_1d(fname: Path, X: np.ndarray) -> None:
    """A strippled down version of 'np.savetxt' for 1D arrays."""
    with fname.open("w") as f:
        rows = [f"{x:.8e}" for x in X]
        out_str = "\n".join(rows)
        f.write(out_str)


def faster_np_savetxt(fname: Path, X: np.ndarray) -> None:
    """A stripped down version of 'np.savetxt' for N-dimensional arrays."""
    _, cols = X.shape
    fmt = ",".join(["%.8e"] * cols)
    with fname.open("w") as f:
        rows = [fmt % tuple(row) for row in X]
        out_str = "\n".join(rows)
        f.write(out_str)