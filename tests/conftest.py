import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import fixture
from fmo_analysis import util

validation_data_dir = Path(__file__).parent.parent / "validation_data"


######## Inputs for validating computations
@fixture
def conf_file():
    return validation_data_dir / "conf.csv"


@fixture
def ham():
    return np.loadtxt(validation_data_dir / "hamiltonian.txt").reshape((7,7))


@fixture
def positions():
    return np.loadtxt(validation_data_dir / "positions.txt").reshape((7,3))


@fixture
def dipole_moments():
    return np.loadtxt(validation_data_dir / "dipole_moments.txt").reshape((7,3))


######## Known good computed data
@fixture
def eigenvalues():
    return np.loadtxt(validation_data_dir / "eigenvalues.txt")


@fixture
def eigenvectors():
    return np.loadtxt(validation_data_dir / "eigenvectors.txt").reshape((7,7))


@fixture
def exciton_dipole_moments():
    return np.loadtxt(validation_data_dir / "exciton_dipole_moments.txt").reshape((7,3))


@fixture
def dipole_strengths():
    return np.loadtxt(validation_data_dir / "dipole_strengths.txt")


@fixture
def rotational_strengths():
    return np.loadtxt(validation_data_dir / "rotational_strengths.txt")


@fixture
def x():
    return np.loadtxt(validation_data_dir / "x.txt")


@fixture
def abs():
    return np.loadtxt(validation_data_dir / "abs.txt")


@fixture
def cd():
    return np.loadtxt(validation_data_dir / "cd.txt")


######## Everything else
@fixture
def config_opts():
    opts = util.DEFAULT_CONFIG.copy()
    # Need to change the number of bandwidths
    # since the validation data has 7 pigments
    opts["abs_bws"] = [120 for _ in range(7)]
    opts["cd_bws"] = [120 for _ in range(7)]
    return opts


@fixture
def config():
    opts = util.DEFAULT_CONFIG.copy()
    # This is the value used to compute the validation data
    opts["bandwidth"] = 120
    opts["abs_bws"] = [120 for _ in range(7)]
    opts["cd_bws"] = [120 for _ in range(7)]
    return util.Config(**opts)


@fixture
def pigments(dipole_moments, positions):
    pigs = list()
    for dpm, r in zip(dipole_moments, positions):
        pigs.append(util.Pigment(r, dpm))
    return pigs


@fixture
def stick_spec(eigenvalues, eigenvectors, exciton_dipole_moments, dipole_strengths, rotational_strengths):
    spec = {
        "e_vals": eigenvalues,
        "e_vecs": eigenvectors,
        "exciton_mus": exciton_dipole_moments,
        "stick_abs": dipole_strengths,
        "stick_cd": rotational_strengths,
    }
    return spec


@fixture
def stick_specs(stick_spec):
    return [stick_spec for _ in range(100)]


@fixture
def broadened_spec(x, abs, cd):
    spec = {
        "x": x,
        "abs": abs,
        "cd": cd,
    }
    return spec


@fixture
def broadened_specs(broadened_spec):
    return [broadened_spec for _ in range(100)]


@fixture
def tempdir():
    with TemporaryDirectory() as td:
        yield Path(td)