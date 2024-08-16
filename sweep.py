import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Job Launcher")
    parser.add_argument("script", type=str, help="Target Python script to execute")
    parser.add_argument(
        "--args", type=str, nargs="+", help="Arguments in the form 'arg1=1,2 arg2=3'"
    )
    parser.add_argument(
        "--gpus", type=str, help="Comma-separated list of available GPUs"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./log", help="Directory to save outputs"
    )
    return parser.parse_args()


def generate_arg_combinations(args_list):
    arg_dict = {}
    for arg in args_list:
        key, values = arg.split("=")
        arg_dict[key] = values.split(",")

    keys = arg_dict.keys()
    combinations = list(itertools.product(*arg_dict.values()))

    return [dict(zip(keys, combination)) for combination in combinations]


def schedule(gpu_ids, jobs, script, sweep_dir):
    gpu_status = {gpu: None for gpu in gpu_ids}
    job_queue = jobs[:]

    run_count = 1
    while job_queue or any(gpu_status.values()):
        for gpu, process in gpu_status.items():
            if process is None and job_queue:
                args = job_queue.pop(0)
                run_dir = os.path.join(sweep_dir, f"run_{run_count}")
                os.makedirs(run_dir, exist_ok=True)
                cmd = ["python", script]
                for k, v in args.items():
                    cmd += [f"--{k}"]
                    cmd += [str(v)]

                print(f"Starting job {run_count} on GPU {gpu}\n{' '.join(cmd)}")

                # Start the job
                process = subprocess.Popen(
                    cmd,
                    stdout=open(os.path.join(run_dir, "output.log"), "w"),
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                )

                # Save args.json in the corresponding run directory
                with open(os.path.join(run_dir, "args.json"), "w") as f:
                    json.dump(args, f, indent=4)

                gpu_status[gpu] = process
                run_count += 1
            elif process is not None and process.poll() is not None:
                gpu_status[gpu] = None
                print(f"Finished job on GPU {gpu}")

        time.sleep(1)


def main():
    args = parse_args()
    gpu_ids = (
        args.gpus.split(",") if args.gpus else ["0"]
    )  # Default to GPU 0 if none specified
    job_args = generate_arg_combinations(args.args)

    # Create the output directory with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = os.path.join(args.output_dir, f"sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    # Save sweep metadata to sweep.json
    sweep_metadata = {
        "script": args.script,
        "args": args.args,
        "gpus": args.gpus,
        "sweep_dir": sweep_dir,
        "job_count": len(job_args),
        "timestamp": timestamp,
    }
    with open(os.path.join(sweep_dir, "sweep.json"), "w") as meta_file:
        json.dump(sweep_metadata, meta_file, indent=4)

    print(f"Generated {len(job_args)} job combinations.")
    schedule(gpu_ids, job_args, args.script, sweep_dir)


if __name__ == "__main__":
    main()
