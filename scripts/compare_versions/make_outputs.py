import os
import subprocess
from pathlib import Path


def main():
    here = Path.cwd()
    dirs = ["known_working", "numpy_optimized", "current"]
    confs_dir = here / "confs"
    for d in dirs:
        case_dir = here / d
        os.chdir(case_dir)
        args = [
            "poetry",
            "run",
            "fmo-analysis",
            "conf2spec",
            "-i",
            confs_dir,
            "-o",
            case_dir / "output",
            "-s",
            "--overwrite"]
        subprocess.run(args)


if __name__ == "__main__":
    main()