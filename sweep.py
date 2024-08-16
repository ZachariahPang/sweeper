import argparse
import itertools
import os
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Job Launcher")
    parser.add_argument("script", type=str, help="Target Python script to execute")
    parser.add_argument(
        "--args", type=str, nargs="+", help="Arguments in the form 'arg1=1,2 arg2=3'"
    )
    parser.add_argument(
        "--gpus", type=str, help="Comma-separated list of available GPUs"
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


def run_job(gpu_id, script, args):
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script} " + " ".join(
        [f"--{k} {v}" for k, v in args.items()]
    )
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)


def schedule(gpu_ids, jobs, script):
    gpu_status = {gpu: None for gpu in gpu_ids}
    job_queue = jobs[:]

    while job_queue or any(gpu_status.values()):
        for gpu, process in gpu_status.items():
            if process is None and job_queue:
                args = job_queue.pop(0)
                cmd = ["python", script] 
                for k, v in args.items():
                    cmd += [f"--{k}"]
                    cmd += [str(v)]
                process = subprocess.Popen(
                    cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                )
                gpu_status[gpu] = process
                print(f"GPU {gpu}: {' '.join(cmd)}")
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

    print(f"Generated {len(job_args)} job combinations.")
    schedule(gpu_ids, job_args, args.script)


if __name__ == "__main__":
    main()
