# Poorman's Sweeper Helper for Multi-gpu Workstations


# Usage

## Running a Sweep
Specify the script to train `toy.py` and the range of parameters to explore:

```{bash}
python sweep.py toy.py --args lr=0.01,0.1 n_layers=2,4,6 --gpus 0,1
```

This command runs `toy.py` with all combinations of learning rates (0.01, 0.1) and layers (2, 4, 6), distributing the jobs across GPUs 0 and 1.

## Example Output
```{bash}
==================================================
6 jobs to be run.

1: python toy.py --lr 0.01 --n_layers 2
2: python toy.py --lr 0.01 --n_layers 4
3: python toy.py --lr 0.01 --n_layers 6
4: python toy.py --lr 0.1 --n_layers 2
5: python toy.py --lr 0.1 --n_layers 4
6: python toy.py --lr 0.1 --n_layers 6
==================================================
```
## Dry Run
To view job combinations without executing them, use:

```{bash}
python sweep.py toy.py --args lr=0.01,0.1 n_layers=2,4,6 --dry_run
```
## Output Logs and Metadata
Each job creates a unique directory containing the metadata and the redirected stdout and stderr
