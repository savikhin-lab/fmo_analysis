from timeit import timeit
from time import perf_counter
from pathlib import Path
from fmo_analysis import util, exciton
from scipy.linalg import lapack
import numpy as np


def main():
    ham, pigs = util.parse_conf_file(Path.cwd() / "conf_brixner_wt" / "conf_brixner_wt.csv")
    config = util.Config(**util.DEFAULT_CONFIG)
    n = 100_000
    times = []
    for _ in range(n):
        t_start = perf_counter()
        _ = exciton.make_fast_stick_spectrum(config, ham, pigs)
        t_stop = perf_counter()
        times.append(t_stop - t_start)
    per_call = sum(times) / n * 1e3
    print(f"{per_call:.4f}ms per call")



if __name__ == "__main__":
    main()