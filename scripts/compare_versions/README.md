# Compare versions
I used these scripts along with git worktree to compare the outputs of different versions of `fmo_analysis`.

The directory structure the scripts expect looks like this:
```
make_outputs.py
compare_outputs.py
confs/
case1/
case2/
case3/
...
```
where `confs` is a directory of `conf` files from YB (e.g. his WT data), and `case1`, etc are the worktrees that you want to compare i.e. different snapshots of `fmo_analysis`. The directory names are hardcoded into the scripts, so you'll need to change them if you use this again.