import click
import json
import matplotlib.pyplot as plt
from pathlib import Path
from . import structures
from . import exciton
from . import util
from .util import Config


@click.group()
@click.version_option()
def cli():
    pass


@click.command()
@click.option("-c", "--config", "config_file", type=click.File(), help="A config file used to override default values in the analysis. See 'default-config' for the default values.")
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path), help="The directory containing the 'conf*.csv' files.")
@click.option("-o", "--output-dir", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path), help="The directory in which the analysis results will be stored.")
@click.option("--overwrite", is_flag=True, default=False, help="If specified, overwrite the data in the output directory.")
@click.option("-b", "--bandwidth", type=click.FLOAT, help="The bandwidth for each transition.")
@click.option("-d", "--delete-pigment", type=click.INT, help="The pigment to delete (0 means none).")
@click.option("-f", "--save-figs", default=False, is_flag=True, help="Save intermediate spectra. An average spectrum is still saved when this flag is not specified.")
@click.option("-s", "--save-intermediate", default=False, is_flag=True, help="Save intermediate results as CSVs")
@click.option("-e", "--empirical", is_flag=True, help="The Hamiltonian is empirical, so don't apply diagonal shifts.")
@click.option("-n", "--normalize", is_flag=True, help="Normalize the total dipole strength to 1")
def conf2spec(config_file, input_dir, output_dir, overwrite, bandwidth, delete_pigment, save_figs, save_intermediate, empirical, normalize):
    # Making sure we have a valid configuration
    if config_file and any([bandwidth, (delete_pigment is not None)]):
        click.echo("Supply config options as flags or in config file, but not both.", err=True)
        return
    if config_file:
        config_opts = util.merge_configs(json.load(config_file))
    else:
        config_opts = util.DEFAULT_CONFIG
    if overwrite:
        config_opts["overwrite"] = overwrite
    if bandwidth:
        config_opts["bandwidth"] = bandwidth
    if delete_pigment is not None:
        config_opts["delete_pig"] = delete_pigment
    if save_figs:
        config_opts["save_figs"] = save_figs
    if save_intermediate:
        config_opts["save_intermediate"] = save_intermediate
    if empirical:
        config_opts["empirical"] = empirical
    if not util.valid_config(config_opts):
        click.echo("Invalid config", err=True)
        return
    config = Config(**config_opts)
    # Loading the Hamiltonians and pigments from disk
    conf_files = util.find_conf_files(input_dir)
    if config.use_shift_T:
        shift_files = util.find_shift_files(conf_files, config)
    # Make sure we're not trying to overwrite anything unless we want to
    if not config.overwrite and util.would_overwrite(output_dir):
        click.echo(f"Attempting to overwrite data in '{output_dir.name}' without specifying '--overwrite'. Exiting.", err=True)
        return
    output_dir.mkdir(exist_ok=True)
    util.save_config(output_dir, config_opts)
    # Compute and save stick spectra
    stick_spectra = exciton.make_stick_spectra(config, conf_files)
    if config.save_intermediate:
        exciton.save_stick_spectra(output_dir, stick_spectra)
    # Compute and save broadened spectra
    broadened_spectra = exciton.make_broadened_spectra(config, stick_spectra)
    exciton.save_broadened_spectra(config, output_dir, broadened_spectra)


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path), help="The directory containing the 'conf*.csv' files.")
@click.option("-o", "--output-dir", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path), help="The directory in which the aligned structures will be saved.")
@click.option("--overwrite", is_flag=True, default=False, help="If specified, overwrite the data in the output directory.")
@click.option("--iterations", "iter", default=100, type=click.INT, help="The maximum number of iterations when optimizing the orientation of each structure.")
@click.option("--tolerance", "tol", default=1e-8, type=click.FLOAT, help="The tolerance maximum RMS difference between the rotated and average structure.")
def align(input_dir, output_dir, overwrite, iter, tol):
    """Rotate the structures for the best alignment between all of them.
    
    The structures may all be rotated relative to one another, which doesn't change
    the physics, but does change the analysis. This command rotates and aligns the
    positions and orientations of the pigments so that the analysis is performed on
    pigments with common alignments."""
    if not overwrite and output_dir.exists():
        click.echo(f"Attempting to overwrite data in '{output_dir.name}' without specifying '--overwrite'. Exiting.", err=True)
        return
    if tol < 0:
        click.echo(f"Tolerance must be >0 but was given '{tol}'. Exiting.", err=True)
        return
    if iter < 1:
        click.echo(f"Iterations must be >1 but was given '{iter}'. Exiting.", err=True)
        return
    output_dir.mkdir(exist_ok=True)
    conf_files = util.find_conf_files(input_dir)
    parsed_confs = [util.parse_conf_file(c) for c in conf_files]
    hams, coords, mus = structures.confs_to_arrs(parsed_confs)
    centered_coords = structures.center_structures(coords)
    rotated_coords, rotated_mus = structures.rotate(centered_coords, mus, iter, tol)
    util.save_conf_files(output_dir, [c.name for c in conf_files], hams, rotated_coords, rotated_mus)


@click.command()
@click.option("-i", "--input-dir", required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path), help="The directory containing the 'conf*.csv' files.")
@click.option("-o", "--output-file", required=False, type=click.Path(dir_okay=False, file_okay=True, path_type=Path), help="The filename when saving the plot.")
@click.option("-m", "--marker", default="dots", type=click.Choice(["dots", "lines"]), help="Whether to display positions as independent dots, or connect them with lines.")
@click.option("-s", "--save", is_flag=True, help="Save the plot rather than displaying it.")
def pigviz(input_dir, output_file, marker, save):
    """Plot the positions of all pigments to inspect alignment.
    
    Waits for the user to press 'Enter' before plotting the next set of pigments."""
    conf_files = util.find_conf_files(input_dir)
    parsed_confs = [util.parse_conf_file(c) for c in conf_files]
    _, coords, _ = structures.confs_to_arrs(parsed_confs)
    fig = plt.figure()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(projection="3d")
    n_confs, _, _ = coords.shape
    for i in range(n_confs):
        xs = coords[i, :, 0]
        ys = coords[i, :, 1]
        zs = coords[i, :, 2]
        if marker == "dots":
            ax.scatter(xs, ys, zs)
        elif marker == "lines":
            ax.plot(xs, ys, zs)
        else:
            click.echo(f"Unknown marker type '{marker}'. Exiting.", err=True)
            return
    if save:
        fig.savefig(output_file, dpi=300)
    else:
        fig.show()
        input("Press Enter to exit.")


@click.command()
def default_config():
    """Prints the default configuration for analysis."""
    print(json.dumps(util.DEFAULT_CONFIG, indent=2, separators=(",", ": ")))


cli.add_command(conf2spec)
cli.add_command(align)
cli.add_command(pigviz)
cli.add_command(default_config)