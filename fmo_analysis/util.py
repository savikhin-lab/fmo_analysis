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
    pignums: int
    delete_pig8: bool
    dip_cor: float
    delete_pig: int
    use_shift_T: bool
    scale: bool
    ignore_offdiagonal_shifts: bool
    overwrite: bool
    save_figs: bool
    save_intermediate: bool


@dataclass
class Pigment:
    pos: np.ndarray
    mu: np.ndarray


def parse_conf_file(config: Config, cf_path: Path) -> Tuple[np.ndarray, List[Pigment]]:
    """Extract the Hamiltonian and pigment data from a 'conf*.csv' file."""
    n_pigs = config.pignums
    arr = np.loadtxt(cf_path)
    rows, cols = arr.shape
    if (rows != n_pigs) or (cols != n_pigs + 6):
        raise ValueError(f"Expected conf file with dimensions {n_pigs}x{n_pigs + 6}, found {rows}x{cols}")
    ham = arr[:, :n_pigs]
    # Shift diagonal elements according to config
    for i in range(n_pigs):
        ham[i, i] += config.shift_diag
    mus = arr[:, -6:-3]
    coords = arr[:, -3:]
    pigments = [Pigment(np.array(c), np.array(m)) for c, m in zip(coords, mus)]
    return ham, pigments


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