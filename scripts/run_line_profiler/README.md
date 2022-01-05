# Run line_profiler

`line_profiler` is a profiler that will show you the breakdown by line of time spent in a function. This script is structured to let you target a specific function in `fmo_analysis`. The function you're profiling is set with the line
```
wrapped = lp(exciton.make_stick_spectrum)
```
Then the report is generated with
```
result = subprocess.run(["python", "-m", "line_profiler", prof_file], capture_output=True, text=True)
```
and saved to the file
```
report_file = Path.cwd() / "lprof_report.txt"
```