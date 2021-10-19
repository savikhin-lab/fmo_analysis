from dataclasses import dataclass
import numpy as np


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