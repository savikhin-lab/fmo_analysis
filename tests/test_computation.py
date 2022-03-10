from fmo_analysis import exciton, util
import numpy.testing as npt
from pytest import raises
from numpy.random import random


def test_can_construct_config():
    c = util.Config(**util.DEFAULT_CONFIG)
    assert c is not None


def test_invalid_config():
    opts = util.DEFAULT_CONFIG.copy()
    opts["bandwidth"] = -1
    assert not util.valid_config(opts)


def test_diagonalizes_hamiltonian(ham, eigenvalues, eigenvectors, pigments, config):
    stick = exciton.stick_spectrum(config, ham, pigments)
    npt.assert_array_almost_equal(eigenvalues, stick["e_vals"], decimal=2)
    computed_vecs = stick["e_vecs"]
    for i in range(7):
        try:
            # Eigenvectors are only defined up to a sign, and sometimes the sign can
            # flip as a result of intermediate calculations during diagonalization
            npt.assert_array_almost_equal(eigenvectors[:, i], computed_vecs[:, i], decimal=4)
        except AssertionError:
            npt.assert_array_almost_equal(eigenvectors[:, i], -computed_vecs[:, i], decimal=4)


def test_computes_exciton_dipole_moments(ham, pigments, config, exciton_dipole_moments):
    stick = exciton.stick_spectrum(config, ham, pigments)
    for i in range(7):
        try:
            npt.assert_array_almost_equal(exciton_dipole_moments[i], stick["exciton_mus"][i], decimal=4)
        except AssertionError:
            npt.assert_array_almost_equal(exciton_dipole_moments[i], -stick["exciton_mus"][i], decimal=4)


def test_computes_dipole_strengths(ham, pigments, config, dipole_strengths):
    stick = exciton.stick_spectrum(config, ham, pigments)
    npt.assert_array_almost_equal(dipole_strengths, stick["stick_abs"], decimal=4)


def test_computes_rotational_strengths(ham, pigments, config, rotational_strengths):
    stick = exciton.stick_spectrum(config, ham, pigments)
    npt.assert_array_almost_equal(rotational_strengths, stick["stick_cd"], decimal=4)


def test_computes_stick_spectra(config, ham, pigments, dipole_strengths, rotational_strengths):
    confs = [{"ham": ham, "pigs": pigments} for _ in range(100)]
    sticks = exciton.stick_spectra(config, confs)
    for s in sticks:
        npt.assert_array_almost_equal(dipole_strengths, s["stick_abs"], decimal=4)
        npt.assert_array_almost_equal(rotational_strengths, s["stick_cd"], decimal=4)


def test_computes_broadened_spectrum_from_ham(config, ham, pigments, abs, cd, x):
    broadened = exciton.broadened_spectrum_from_conf(config, {"ham": ham, "pigs": pigments})
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)


def test_computes_broadened_spectrum_from_stick(config, dipole_strengths, rotational_strengths, eigenvalues, abs, cd, x):
    mock_stick_spec = {
        "stick_abs": dipole_strengths,
        "stick_cd": rotational_strengths,
        "e_vals": eigenvalues
    }
    broadened = exciton.broadened_spectrum_from_stick(config, mock_stick_spec)
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)


def test_computes_broadened_spectra_from_hams(config, ham, pigments, abs, cd, x):
    confs = [{"ham": ham, "pigs": pigments} for _ in range(100)]
    broadened = exciton.broadened_spectrum_from_confs(config, confs)
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)


def test_computes_avg_spec(broadened_spec):
    specs = [broadened_spec for _ in range(10)]
    avg = exciton.average_broadened_spectra(specs)
    npt.assert_almost_equal(broadened_spec["x"], avg["x"], decimal=4)
    npt.assert_almost_equal(broadened_spec["abs"], avg["abs"], decimal=4)
    npt.assert_almost_equal(broadened_spec["cd"], avg["cd"], decimal=4)


def test_stick_spec_deletes_pigment(config_opts, ham, pigments, dipole_strengths):
    config_opts["delete_pig"] = 1
    config = util.Config(**config_opts)
    spec = exciton.stick_spectrum(config, ham, pigments)
    with raises(AssertionError):
        npt.assert_almost_equal(dipole_strengths, spec["stick_abs"], decimal=4)


def test_stick_spectra_deletes_pigment(config_opts, ham, pigments, dipole_strengths):
    config_opts["delete_pig"] = 1
    config = util.Config(**config_opts)
    n_confs = 10
    confs = [{"ham": ham, "pigs": pigments} for _ in range(n_confs)]
    specs = exciton.stick_spectra(config, confs)
    for s in specs:
        with raises(AssertionError):
            npt.assert_almost_equal(dipole_strengths, s["stick_abs"], decimal=4)


def test_broadened_spectrum_deletes_pigment(config_opts, ham, pigments, abs):
    config_opts["delete_pig"] = 1
    config = util.Config(**config_opts)
    spec = exciton.broadened_spectrum_from_conf(config, {"ham": ham, "pigs": pigments})
    with raises(AssertionError):
        npt.assert_almost_equal(abs, spec["abs"], decimal=4)


def test_broadened_spectrum_multi_deletes_pigment(config_opts, ham, pigments, abs):
    config_opts["delete_pig"] = 1
    config = util.Config(**config_opts)
    n_confs = 10
    confs = [{"ham": ham, "pigs": pigments} for _ in range(n_confs)]
    spec = exciton.broadened_spectrum_from_confs(config, confs)
    with raises(AssertionError):
        npt.assert_almost_equal(abs, spec["abs"], decimal=4)


def test_computes_het_broadened_spectrum_from_ham(config, ham, pigments, x, abs, cd):
    broadened = exciton.het_broadened_spectrum_from_conf(config, {"ham": ham, "pigs": pigments})
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)


def test_het_broadened_spectrum_deletes_pigment(config_opts, ham, pigments, abs):
    config_opts["delete_pig"] = 1
    config = util.Config(**config_opts)
    spec = exciton.het_broadened_spectrum_from_conf(config, {"ham": ham, "pigs": pigments})
    with raises(AssertionError):
        npt.assert_almost_equal(abs, spec["abs"], decimal=4)


def test_computes_het_broadened_spectrum_from_stick(config, dipole_strengths, rotational_strengths, eigenvalues, abs, cd, x):
    mock_stick_spec = {
        "stick_abs": dipole_strengths,
        "stick_cd": rotational_strengths,
        "e_vals": eigenvalues
    }
    broadened = exciton.het_broadened_spectrum_from_stick(config, mock_stick_spec)
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)


def test_computes_het_broadened_spectrum_from_hams(config, ham, pigments, abs, cd, x):
    confs = [{"ham": ham, "pigs": pigments} for _ in range(100)]
    broadened = exciton.het_broadened_spectrum_from_confs(config, confs)
    npt.assert_array_almost_equal(x, broadened["x"], decimal=4)
    npt.assert_array_almost_equal(abs, broadened["abs"], decimal=4)
    npt.assert_array_almost_equal(cd, broadened["cd"], decimal=4)


def test_broadened_spec_isnt_readonly(config, stick_spec):
    b_spec = exciton.broadened_spectrum_from_stick(config, stick_spec)
    b_spec["abs"] *= 0


def test_array_to_conf_conversion():
    dummy_hams = random(size=(100, 8, 8))
    dummy_mus = random(size=(100, 8, 3))
    dummy_rs = random(size=(100, 8, 3))
    hams, mus, rs = exciton.confs_to_arrays(exciton.arrays_to_confs(dummy_hams, dummy_mus, dummy_rs))
    npt.assert_array_almost_equal(dummy_hams, hams, decimal=4)
    npt.assert_array_almost_equal(dummy_mus, mus, decimal=4)
    npt.assert_array_almost_equal(dummy_rs, rs, decimal=4)