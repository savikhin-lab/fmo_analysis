import click
import numpy as np
from pathlib import Path


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
def run():
    pass


cli.add_command(run)