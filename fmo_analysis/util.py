from dataclasses import dataclass
import numpy as np


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


@dataclass
class Pigment:
    pos: np.ndarray
    mu: np.ndarray


def parse_conf_file(cf_path, config):
    """Extract the Hamiltonian and pigment data from a 'conf*.csv' file."""
    n_pigs = config["pignums"]
    arr = np.loadtxt(cf_path)
    rows, cols = arr.shape
    if (rows != n_pigs) or (cols != n_pigs + 6):
        raise ValueError(f"Expected conf file with dimensions {n_pigs}x{n_pigs + 6}, found {rows}x{cols}")
    ham = arr[:, :n_pigs]
    mus = arr[:, -6:-3]
    coords = arr[:, -3:]
    pigments = [Pigment(np.array(c), np.array(m)) for c, m in zip(coords, mus)]
    return {"ham": ham, "pigs": pigments}