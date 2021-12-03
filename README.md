# FMO Analysis

This program is a toolbox for computing spectra of the FMO complex.

## Installation
### Poetry
If you have [`poetry`](https://python-poetry.org) installed, just download the source code for this directory, navigate to it in your terminal, and do the following:
```
$ poetry build
$ python -m pip install --user dist/fmo_analysis-X.Y.Z-py3-none-any.whl
```
where `X.Y.Z` is the version number of the package.

### Pip
You can alternatively install the package straight from GitHub using `pip`:
```
$ python -m pip install --user git+git@github.com:savikhin-lab/fmo_analysis.git
```

## Usage - Command Line
You can see which commands are available from the command line:
```
$ fmo-analysis --help
Usage: fmo-analysis [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  align           Rotate the structures for the best alignment between...
  conf2spec       Generate spectra from one or more 'conf*.csv' files.
  default-config  Prints the default configuration for analysis.
  pigviz          Plot the positions of all pigments to inspect alignment.
```

You can get further help for a command via `fmo-analysis <command> --help`, which will display all of the options and flags available to the command:
```
Usage: fmo-analysis conf2spec [OPTIONS]

  Generate spectra from one or more 'conf*.csv' files.

Options:
  -c, --config FILENAME         A config file used to override default values
                                in the analysis. See 'default-config' for the
                                default values.
  -i, --input-dir DIRECTORY     The directory containing the 'conf*.csv'
                                files.  [required]
  -o, --output-dir DIRECTORY    The directory in which the analysis results
                                will be stored.  [required]
  --overwrite                   If specified, overwrite the data in the output
                                directory.
  -b, --bandwidth FLOAT         The bandwidth for each transition.
  -d, --delete-pigment INTEGER  The pigment to delete (0 means none).
  -f, --save-figs               Save intermediate spectra. An average spectrum
                                is still saved when this flag is not
                                specified.
  -s, --save-intermediate       Save intermediate results as CSVs
  -e, --empirical               The Hamiltonian is empirical, so don't apply
                                diagonal shifts.
  -n, --normalize               Normalize the total dipole strength to 1
  --help                        Show this message and exit.
```

## Usage - Package
The most useful modules in this package are `util` and `exciton`.

### `util`
This module handles finding, loading, and parsing of `conf*.csv` files. There are also facilities for saving a Hamiltonian and a set of pigments formatted as a `conf` file. This is useful for instance if you've shifted elements of a Hamiltonian and want to save the shifted `conf` for later use.

This module also contains the set of default configuration options for computing spectra in a dictionary called `util.DEFAULT_CONFIG`. However, all of the functions that compute spectra require a `util.Config` object. What's the difference? A `util.Config` object is read-only, which prevents you from accidentally changing the config while you're computing spectra. To change the configuration you can make a copy of `util.DEFAULT_CONFIG`, set the dictionary entries that you care about, and then create your `util.Config` object.

Note that `util.DEFAULT_CONFIG` includes a constant diagonal shift because the data from YB needs a shift in order to be at the correct wavelength. The empirical Hamiltonians don't need this shift, so it's important to set the `shift_diag` option to 0 when computing spectra from empirical Hamiltonians.

### `exciton`
This module handles computing spectra. In order to compute spectra, you first need a configuration (`util.Config`). This object sets the bandwidth, which pigment to delete (if any), etc.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.