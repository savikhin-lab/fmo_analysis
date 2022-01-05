import numpy as np
import subprocess
from line_profiler import LineProfiler
from fmo_analysis import exciton, util
from pathlib import Path


def main():
    lp = LineProfiler()
    wrapped = lp(exciton.make_stick_spectrum)
    conf_file = Path.cwd() / "conf_brixner_wt" / "conf_brixner_wt.csv"
    ham, pigs = util.parse_conf_file(conf_file)
    config = util.Config(**util.DEFAULT_CONFIG)
    sticks = wrapped(config, ham, pigs)
    np.savetxt(Path.cwd() / "sticks2.txt", sticks["stick_cd"])
    prof_file = Path.cwd() / "out.lprof"
    report_file = Path.cwd() / "lprof_report.txt"
    lp.dump_stats(prof_file)
    result = subprocess.run(["python", "-m", "line_profiler", prof_file], capture_output=True, text=True)
    with report_file.open("w") as f:
        f.write(result.stdout)


if __name__ == "__main__":
    main()