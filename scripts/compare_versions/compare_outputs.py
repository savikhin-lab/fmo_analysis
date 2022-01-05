import subprocess
import numpy as np
from pathlib import Path


dirs = ["known_working", "numpy_optimized", "current"]
n_cases = len(dirs)
here = Path.cwd()
np.set_printoptions(linewidth=150)
tol = 0.01


def main():
    for i in range(1, 101):
        d_name = f"conf{i:03d}"
        e_vals = load_e_vals(d_name)
        e_vecs = load_e_vecs(d_name, e_vals)
        dpm = load_dpm(d_name)
        stick_abs = load_stick_abs(d_name)
        stick_cd = load_stick_cd(d_name)
        # if i == 15:
        #     breakpoint()
        if not e_vals_match(e_vals):
            print(f"Eigenvalues don't match, i={i}")
            return
        if not e_vecs_match(e_vecs):
            print(f"Eigenvectors don't match, i={i}")
            return
        if not dpm_match(dpm):
            print(f"Dipole moments don't match, i={i}")
            return
        if not stick_abs_matches(stick_abs):
            print(f"Abs. doesn't match, i={i}")
            return
        if not stick_cd_matches(stick_cd):
            print(f"CD doesn't match, i={i}")
            return


def load_stick_abs(d):
    data = np.empty((8, 3))
    for i, system in enumerate(dirs):
        fname = here / system / "output" / "stick_spectra" / d / "stick_cd.csv"
        data[:, i] = np.loadtxt(fname, delimiter=",")
    return data


def stick_abs_matches(data):
    diffs = np.empty((8, 2))
    for i in range(8):
        numpy_optimized_rel_err = np.abs(data[i, 0] - data[i, 1]) / np.abs(data[i, 0])
        current_rel_err = np.abs(data[i, 0] - data[i, 2]) / np.abs(data[i, 0])
        diffs[i, 0] = numpy_optimized_rel_err
        diffs[i, 1] = current_rel_err
    diffs_too_large = diffs > 0.01
    if np.any(diffs_too_large):
        return False
    return True


def load_dpm(d):
    data = np.empty((3, 8, 3))
    for i, system in enumerate(dirs):
        fname = here / system / "output" / "stick_spectra" / d / "exciton_mus.csv"
        dpm = np.loadtxt(fname, delimiter=",")
        data[i, :, :] = dpm 
    return data


def dpm_match(data):
    for i in range(8):
        diffs = np.abs(data[1, i, :] - data[0, i, :]) / np.abs(data[0, i, :])
        if np.any(diffs > tol):
            diffs = np.abs(data[1, i, :] + data[0, i, :]) / np.abs(data[0, i, :])
            if np.any(diffs > tol):
                return False
        diffs = np.abs(data[2, i, :] - data[0, i, :]) / np.abs(data[0, i, :])
        if np.any(diffs > tol):
            diffs = np.abs(data[2, i, :] + data[0, i, :]) / np.abs(data[0, i, :])
            if np.any(diffs > tol):
                return False
    return True


def load_e_vecs(d, e_vals):
    data = np.empty((3, 9, 8))  # Eigenvalues in the first row for sorting purposes
    for i, system in enumerate(dirs):
        fname = here / system / "output" / "stick_spectra" / d / "eigenvectors.csv"
        e_vecs = np.loadtxt(fname, delimiter=",")
        data[i, 0, :] = e_vals[:, i]
        data[i, 1:, :] = e_vecs
    return data


def e_vecs_match(data):
    e_vecs = data[:, 1:, :]
    for i in range(8):
        opt_rel_err = np.abs(e_vecs[1, :, i] - e_vecs[0, :, i]) / np.abs(e_vecs[0, :, i])
        if np.any(opt_rel_err > tol):
            # Sometimes the sign of an eigenvector is flipped
            opt_rel_err = np.abs(e_vecs[1, :, i] + e_vecs[0, :, i]) / np.abs(e_vecs[0, :, i])
            if np.any(opt_rel_err > tol):
                return False
        curr_rel_err = np.abs(e_vecs[2, :, i] - e_vecs[0, :, i]) / np.abs(e_vecs[0, :, i])
        if np.any(curr_rel_err > tol):
            # Sometimes the sign of an eigenvector is flipped
            curr_rel_err = np.abs(e_vecs[2, :, i] + e_vecs[0, :, i]) / np.abs(e_vecs[0, :, i])
            if np.any(curr_rel_err > tol):
                return False
    return True


def load_e_vals(d, sort=False):
    data = np.empty((8, 3))
    for i, system in enumerate(dirs):
        fname = here / system / "output" / "stick_spectra" / d / "energies.csv"
        e_vals = np.loadtxt(fname, delimiter=",")
        if sort:
            e_vals = np.array(sorted(e_vals))
        data[:, i] = e_vals
    return data


def e_vals_match(data):
    diffs = np.empty((8, 2))
    for i in range(8):
        numpy_optimized_rel_err = np.abs(data[i, 0] - data[i, 1]) / np.abs(data[i, 0])
        current_rel_err = np.abs(data[i, 0] - data[i, 2]) / np.abs(data[i, 0])
        diffs[i, 0] = numpy_optimized_rel_err
        diffs[i, 1] = current_rel_err
    diffs_too_large = diffs > tol
    if np.any(diffs_too_large):
        return False
    return True


def load_stick_cd(d):
    data = np.empty((8, 3))
    for i, system in enumerate(dirs):
        fname = here / system / "output" / "stick_spectra" / d / "stick_cd.csv"
        data[:, i] = np.loadtxt(fname, delimiter=",")
    return data


def stick_cd_matches(data):
    diffs = np.empty((8, 2))
    for i in range(8):
        numpy_optimized_rel_err = np.abs(data[i, 0] - data[i, 1]) / np.abs(data[i, 0])
        current_rel_err = np.abs(data[i, 0] - data[i, 2]) / np.abs(data[i, 0])
        diffs[i, 0] = numpy_optimized_rel_err
        diffs[i, 1] = current_rel_err
    diffs_too_large = diffs > 0.01
    if np.any(diffs_too_large):
        return False
    return True


if __name__ == "__main__":
    main()