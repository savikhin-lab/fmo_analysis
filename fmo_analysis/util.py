import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_CONFIG = {
    "xfrom": 11790,
    "xto": 13300,
    "xstep": 1,
    "bandwidth": 70,
    "shift_diag": -2420,
    "dip_cor": 0.014,
    "delete_pig": 0,
    "use_shift_T": False,
    "scale": False,
    "overwrite": False,
    "save_figs": False,
    "save_intermediate": False,
    "empirical": False,
    "normalize": False
}


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
    normalize: bool


@dataclass
class Pigment:
    pos: np.ndarray
    mu: np.ndarray


def find_conf_files(input_dir: Path) -> List[Path]:
    """Obtains paths for the 'conf*.csv' files containing the Hamiltonians."""
    conf_files = sorted([f for f in input_dir.iterdir() if f.name[:4] == "conf"])
    if conf_files == []:
        raise FileNotFoundError(f"No 'conf*.csv' files found in directory '{input_dir}'")
    return conf_files


def find_shift_files(files: List[Path], config: Config) -> List[Path]:
    """Obtains paths for the 'conf*-shift.csv' files."""
    parent = files[0].parent
    paths = []
    for f in files:
        stem = f.stem
        ext = f.suffix
        shift_path = parent / (stem + "-shift" + ext)
        if not shift_path.exists():
            raise FileNotFoundError(f"Shift file '{shift_path.name}' not found.")
        paths.append(shift_path)
    return paths


def would_overwrite(outdir: Path) -> bool:
    """Walks the directory structure to make sure nothing exists that would be overwritten."""
    if outdir.exists():
        return True
    stick_dir = outdir / "stick_spectra"
    if stick_dir.exists():
        return True
    broadened_dir = outdir / "broadened_spectra"
    if broadened_dir.exists():
        return True
    return False


def merge_default_config_with_file(user_supplied_config):
    """Updates the default config with user-supplied values."""
    merged = {}
    for k in DEFAULT_CONFIG.keys():
        merged[k] = user_supplied_config.get(k, DEFAULT_CONFIG[k])
    return merged


def valid_config(config: Dict) -> bool:
    """Ensure that any configuration errors are caught before starting analysis."""
    # Make sure some values are numeric
    numeric = [isinstance(config[k], numbers.Number) for k in
        ["xfrom", "xto", "xstep", "bandwidth", "shift_diag", "dip_cor", "delete_pig"]]
    if not all(numeric):
        return False
    bounds_checks = [
        config["xfrom"] > 0,
        config["xto"] > config["xfrom"],
        config["xstep"] > 0,
        config["bandwidth"] > 0,
        0 <= config["delete_pig"] <= 8
    ]
    if not all(bounds_checks):
        return False
    # Make sure some values are boolean
    bool_checks = [isinstance(config[k], bool) for k in
        ["use_shift_T", "scale", "overwrite", "save_figs", "save_intermediate", "empirical", "normalize"]]
    if not all(bool_checks):
        return False
    return True


def save_config(outdir: Path, config: Dict) -> None:
    """Save the config used for this analysis to disk for auditing later."""
    try:
        del config["overwrite"]
    except KeyError:
        pass
    out_path = outdir / "config_used.json"
    with out_path.open("w") as f:
        json.dump(config, f)


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


def load_confs(paths: List[Path]) -> List[Dict]:
    """Load and parse the supplied conf files"""
    parsed = list()
    for p in paths:
        ham, pigs = parse_conf_file(p)
        parsed.append({"ham": ham, "pigs": pigs, "file": p})
    return parsed


def save_conf_files(outdir: Path, filenames: List[str], hams: np.ndarray, coords: np.ndarray, mus: np.ndarray) -> None:
    """Save conf files from arrays."""
    _, n_pigs, _ = hams.shape
    for i, fname in enumerate(filenames):
        out_arr = np.zeros((n_pigs, n_pigs + 6))
        out_arr[:, :n_pigs] = hams[i, :, :]
        out_arr[:, -6:-3] = mus[i, :, :]
        out_arr[:, -3:] = coords[i, :, :]
        faster_np_savetxt(outdir / fname, out_arr)


def assemble_conf_file(ham: np.ndarray, pigs: List[Pigment]) -> np.ndarray:
    """Stuffs a Hamiltonian and pigment info into a single array to be saved later."""
    n_pigs, _ = ham.shape
    arr = np.zeros((n_pigs, n_pigs + 6))
    arr[:, :n_pigs] = ham
    for i, p in enumerate(pigs):
        arr[i, -6:-3] = p.mu
        arr[i, -3:] = p.pos
    return arr


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
