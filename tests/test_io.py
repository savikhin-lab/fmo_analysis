import numpy.testing as npt
from pathlib import Path
from fmo_analysis import exciton, util

######## Finding and loading conf files
def test_finds_conf_files(tempdir):
    n_confs = 10
    for i in range(n_confs):
        c_name = tempdir / f"conf{i}.csv"
        with c_name.open("w") as f:
            f.write("")
    confs_found = util.find_conf_files(tempdir)
    assert len(confs_found) == n_confs


def test_loads_conf(conf_file, ham, dipole_moments, positions):
    parsed_ham, parsed_pigs = util.parse_conf_file(conf_file)
    npt.assert_array_almost_equal(ham, parsed_ham, decimal=2)
    assert parsed_ham.shape == (7, 7)
    assert len(parsed_pigs) == 7
    for i, p in enumerate(parsed_pigs):
        npt.assert_array_almost_equal(positions[i], parsed_pigs[i].pos, decimal=4)
        npt.assert_array_almost_equal(dipole_moments[i], parsed_pigs[i].mu, decimal=4)


######## Saving spectra
def test_saves_single_stick_spectrum(tempdir, stick_spec):
    stick_spec["file"] = "conf.csv"
    exciton.save_stick_spectrum(tempdir, stick_spec)
    for name in ["energies", "eigenvectors", "exciton_mus", "stick_abs", "stick_cd"]:
        fname = tempdir / f"{name}.csv"
        assert fname.exists()


def test_saves_multiple_stick_spectra(tempdir, stick_spec):
    n_specs = 10
    specs = [stick_spec.copy() for _ in range(n_specs)]
    for i, s in enumerate(specs):
        s["file"] = Path(f"conf{i}.csv")
    exciton.save_stick_spectra(tempdir, specs)
    spec_dirs = sorted([x for x in (tempdir / "stick_spectra").iterdir()])
    for i in range(n_specs):
        assert spec_dirs[i].name == f"conf{i}"


def test_saves_single_broadened_spectrum(tempdir, broadened_spec):
    exciton.save_broadened_spectrum(tempdir, broadened_spec)
    for name in ["abs", "cd"]:
        fname = tempdir / f"{name}.csv"
        assert fname.exists()
    for units in ["wn", "nm"]:
        fname = tempdir / f"spec_{units}.tiff"
        assert fname.exists()


def test_saves_multiple_broadened_spectra(tempdir, broadened_spec, config_opts):
    n_specs = 10
    specs = [broadened_spec.copy() for _ in range(n_specs)]
    for i, s in enumerate(specs):
        s["file"] = Path(f"conf{i}.csv")
    config_opts["save_figs"] = True
    config = util.Config(**config_opts)
    exciton.save_broadened_spectra(config, tempdir, specs)
    abs_dir = tempdir / "broadened_spectra" / "abs"
    assert abs_dir.exists()
    cd_dir = tempdir / "broadened_spectra" / "cd"
    assert cd_dir.exists()
    plots_dir = tempdir / "broadened_spectra" / "plots"
    assert plots_dir.exists()
    for i in range(n_specs):
        abs_file = abs_dir / f"conf{i}_abs.csv"
        assert abs_file.exists()
        cd_file = cd_dir / f"conf{i}_cd.csv"
        assert cd_file.exists()
        plot_file = plots_dir / f"conf{i}.tiff"
        assert plot_file.exists()
