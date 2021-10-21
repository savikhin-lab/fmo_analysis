import click
import json
import numbers
import numpy as np
from pathlib import Path
from typing import List
from . import exciton
from .util import Config, parse_conf_file, save_stick_spectrum


DEFAULT_CONFIG = {
    "xfrom": 11790,
    "xto": 13300,
    "xstep": 1,
    "bandwidth": 200,
    "shift_diag": -2420,
    "pignums": 8,
    "delete_pig8": False,
    "dip_cor": 0.014,
    "delete_pig": 0,
    "use_shift_T": False,
    "scale": False,
    "ignore_offdiagonal_shifts": False,
    "overwrite": False,
}


@click.group()
def cli():
    pass


@click.command()
@click.option("-c", "--config", "config_file", type=click.File(), help="A config file used to override default values in the analysis. See 'default-config' for the default values.")
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path), help="The directory containing the 'conf*.csv' files.")
@click.option("-o", "--output-dir", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path), help="The directory in which the analysis results will be stored.")
@click.option("--overwrite", is_flag=True, default=False, help="If specified, overwrite the data in the output directory.")
def run(config_file, input_dir, output_dir, overwrite):
    # Making sure we have a valid configuration
    if config_file:
        config_opts = merge_configs(json.load(config_file))
    else:
        config_opts = DEFAULT_CONFIG
    if overwrite:
        config_opts["overwrite"] = overwrite
    if not valid_config(config_opts):
        click.echo("Invalid config", err=True)
        return
    config = Config(**config_opts)
    # Loading the Hamiltonians and pigments from disk
    conf_files = find_conf_files(input_dir)
    if config.use_shift_T:
        shift_files = find_shift_files(conf_files, config)
    # Make sure we're not trying to overwrite anything unless we want to
    if not config.overwrite and would_overwrite(output_dir):
        click.echo(f"Attempting to overwrite data in '{output_dir.name}' without specifying '--overwrite'. Exiting.", err=True)
        return
    output_dir.mkdir(exist_ok=True)
    sticks_dir = output_dir / "stick_spectra"
    sticks_dir.mkdir(exist_ok=True)
    stick_results = []
    for c in conf_files:
        ham, pigs = parse_conf_file(config, c)
        stick_result = exciton.make_stick_spectra(config, ham, pigs)
        stick_result["file"] = c
        stick_results.append(stick_result)
        save_stick_spectrum(sticks_dir, stick_result, overwrite)
    

@click.command()
def default_config():
    """Prints the default configuration for analysis."""
    print(json.dumps(DEFAULT_CONFIG, indent=2, separators=(",", ": ")))


def merge_configs(user_supplied_config):
    """Updates the default config with user-supplied values."""
    merged = {}
    for k in DEFAULT_CONFIG.keys():
        merged[k] = user_supplied_config.get(k, DEFAULT_CONFIG[k])
    return merged


def valid_config(config):
    """Ensure that any configuration errors are caught before starting analysis."""
    # Make sure some values are numeric
    numeric = [isinstance(config[k], numbers.Number) for k in
        ["xfrom", "xto", "xstep", "bandwidth", "shift_diag", "pignums", "dip_cor", "delete_pig"]]
    if not all(numeric):
        return False
    bounds_checks = [
        config["xfrom"] > 0,
        config["xto"] > config["xfrom"],
        config["xstep"] > 0,
        config["bandwidth"] > 0,
        config["pignums"] > 0,
        0 <= config["delete_pig"] < config["pignums"]
    ]
    if not all(bounds_checks):
        return False
    # Make sure some values are boolean
    bool_checks = [isinstance(config[k], bool) for k in
        ["delete_pig8", "use_shift_T", "scale", "ignore_offdiagonal_shifts", "overwrite"]]
    if not all(bool_checks):
        return False
    return True


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

cli.add_command(run)
cli.add_command(default_config)