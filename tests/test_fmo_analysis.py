from fmo_analysis import exciton, util
from pathlib import Path
import numpy as np
import numpy.testing as npt
from pytest import fixture


validation_data_dir = Path(__file__).parent.parent / "validation_data"


@fixture
def ham():
    return np.loadtxt(validation_data_dir / "hamiltonian.txt").reshape((7,7))


@fixture
def positions():
    return np.loadtxt(validation_data_dir / "positions.txt").reshape((7,3))


@fixture
def dipole_moments():
    return np.loadtxt(validation_data_dir / "dipole_moments.txt").reshape((7,3))


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


@fixture
def config():
    opts = util.DEFAULT_CONFIG.copy()
    opts["bandwidth"] = 120
    return util.Config(**opts)


@fixture
def pigments(dipole_moments, positions):
    pigs = list()
    for dpm, r in zip(dipole_moments, positions):
        pigs.append(util.Pigment(r, dpm))
    return pigs


def test_diagonalizes_hamiltonian(ham, eigenvalues, eigenvectors, pigments, config):
    stick = exciton.make_stick_spectrum(config, ham, pigments)
    npt.assert_array_almost_equal(eigenvalues, stick["e_vals"], decimal=2)
    computed_vecs = stick["e_vecs"]
    for i in range(7):
        try:
            # Eigenvectors are only defined up to a sign, and sometimes the sign can
            # flip based on precision, etc
            npt.assert_array_almost_equal(eigenvectors[:, i], computed_vecs[:, i], decimal=4)
        except AssertionError:
            npt.assert_array_almost_equal(eigenvectors[:, i], -computed_vecs[:, i], decimal=4)


def test_computes_exciton_dipole_moments(ham, pigments, config, exciton_dipole_moments):
    stick = exciton.make_stick_spectrum(config, ham, pigments)
    for i in range(7):
        try:
            npt.assert_array_almost_equal(exciton_dipole_moments[i], stick["exciton_mus"][i], decimal=4)
        except AssertionError:
            npt.assert_array_almost_equal(exciton_dipole_moments[i], -stick["exciton_mus"][i], decimal=4)


def test_computes_dipole_strengths(ham, pigments, config, dipole_strengths):
    stick = exciton.make_stick_spectrum(config, ham, pigments)
    npt.assert_array_almost_equal(dipole_strengths, stick["stick_abs"], decimal=4)


def test_computes_rotational_strengths(ham, pigments, config, rotational_strengths):
    stick = exciton.make_stick_spectrum(config, ham, pigments)
    npt.assert_array_almost_equal(rotational_strengths, stick["stick_cd"], decimal=4)


def test_computes_broadened_spectrum(config, dipole_strengths, rotational_strengths, eigenvalues, abs, cd, x):
    mock_stick_spec = {
        "stick_abs": dipole_strengths,
        "stick_cd": rotational_strengths,
        "e_vals": eigenvalues
    }
    broadened = exciton.make_broadened_spectrum(config, mock_stick_spec)
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)