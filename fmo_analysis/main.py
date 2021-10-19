import click
import json
import numbers


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
    "ignore_offdiagonal_shifts": False
}


@click.group()
def cli():
    pass


@click.command()
@click.option("-c", "--config", "config_file", type=click.File(), help="A config file used to override default values in the analysis. See 'default-config' for the default values.")
def run(config_file):
    if config_file:
        config = merge_configs(json.load(config_file))
    else:
        config = DEFAULT_CONFIG
    if not valid_config(config):
        click.echo("Invalid config", err=True)
        return


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
    bool_checks = [isinstance(config[k], bool) for k in
        ["delete_pig8", "use_shift_T", "scale", "ignore_offdiagonal_shifts"]]
    if not all(bool_checks):
        return False
    return True

cli.add_command(run)
cli.add_command(default_config)