import argparse
import itertools
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional


def parse_args():
    """
    Parses command-line arguments for the Deep Learning Job Launcher.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
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
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print metadata without executing the jobs",
    )
    return parser.parse_args()


def generate_cmd_combinations(script: str, args_list: List[str]) -> List[List[str]]:
    """
    Generates a list of command-line argument combinations for the specified script.

    Args:
        script (str): The script to be executed.
        args_list (List[str]): A list of argument strings in the form 'arg1=1,2 arg2=3'.
        For boolean arguments, use "true" or "false" as values.
        - Boolean values (i.e., "true" or "false") are handled specially:
          - If the value is "true", the argument is added as a flag (e.g., "--arg").
          - If the value is "false", the argument is added as a negated flag (e.g., "--no-arg").

    Returns:
        List[List[str]]: A list of command-line commands with all argument combinations.

    """
    arg_dict: Dict[str, List[str]] = {}
    for arg in args_list:
        key, values = arg.split("=")
        arg_dict[key] = values.split(",")

    keys = arg_dict.keys()
    combinations = list(itertools.product(*arg_dict.values()))

    cmd_list: List[List[str]] = []
    for combination in combinations:
        cmd = ["python", script]
        for key, value in zip(keys, combination):
            if value.lower() == "true":
                cmd += [f"--{key}"]
            elif value.lower() == "false":
                cmd += [f"--no-{key}"]
            else:
                cmd += [f"--{key}", str(value)]
        cmd_list.append(cmd)

    return cmd_list


def schedule(
    gpu_ids: List[str], jobs: List[List[str]], sweep_dir: str
) -> List[Dict[str, Optional[int]]]:
    """
    Schedules and executes a list of jobs on available GPUs.

    Args:
        gpu_ids (List[str]): List of available GPU IDs.
        jobs (List[List[str]]): List of command-line commands to execute.
        sweep_dir (str): Directory to store logs and metadata.

    Returns:
        List[Dict[str, Optional[int]]]: List of failed jobs with details.
    """
    gpu_status: Dict[str, Optional[subprocess.Popen]] = {gpu: None for gpu in gpu_ids}
    job_queue = jobs[:]
    failed_jobs: List[Dict[str, Optional[int]]] = []

    run_count = 1
    while job_queue or any(gpu_status.values()):
        for gpu, process in gpu_status.items():
            if process is None and job_queue:
                cmd = job_queue.pop(0)
                run_dir = os.path.join(sweep_dir, f"run_{run_count}")
                os.makedirs(run_dir, exist_ok=True)

                print(f"Running job {run_count} on GPU {gpu}: {' '.join(cmd)}")

                # Start the job
                process = subprocess.Popen(
                    cmd,
                    stdout=open(os.path.join(run_dir, "output.log"), "w"),
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                )

                # Save cmd and run directory in the process object
                gpu_status[gpu] = process
                gpu_status[gpu].run_dir = run_dir
                gpu_status[gpu].cmd = cmd
                run_count += 1
            elif process is not None and process.poll() is not None:
                if process.returncode != 0:
                    print(
                        f"Job on GPU {gpu} failed with exit code {process.returncode}"
                    )
                    print(
                        f"Check the logs at {gpu_status[gpu].run_dir}/output.log for more details."
                    )
                    failed_jobs.append(
                        {
                            "gpu": gpu,
                            "run_dir": gpu_status[gpu].run_dir,
                            "cmd": gpu_status[gpu].cmd,
                            "exit_code": process.returncode,
                        }
                    )
                else:
                    print(f"Job on GPU {gpu} finished.")
                gpu_status[gpu] = None

        time.sleep(1)

    return failed_jobs


def main() -> None:
    """
    Main function to parse arguments, generate job combinations,
    and schedule the jobs for execution.
    """
    args = parse_args()
    gpu_ids = (
        args.gpus.split(",") if args.gpus else ["0"]
    )  # Default to GPU 0 if none specified
    job_cmds = generate_cmd_combinations(args.script, args.args)

    print("=" * 50)
    print(f"{len(job_cmds)} jobs to be run.\n")
    for i, cmd in enumerate(job_cmds):
        print(f"{i+1}: " + " ".join(cmd))
    print("=" * 50 + "\n")

    if not args.dry_run:
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
            "job_count": len(job_cmds),
            "jobs": job_cmds,
            "timestamp": timestamp,
        }
        with open(os.path.join(sweep_dir, "sweep.json"), "w") as meta_file:
            json.dump(sweep_metadata, meta_file, indent=4)

        failed_jobs = schedule(gpu_ids, job_cmds, sweep_dir)

        if failed_jobs:
            print("\n", "=" * 50 + " Summary of Failed Jobs:")
            for job in failed_jobs:
                print(
                    f"GPU: {job['gpu']}, Run Directory: {job['run_dir']}, Exit Code: {job['exit_code']}"
                )
                print(f"Command: {' '.join(job['cmd'])}")
        else:
            print("All jobs completed successfully.")


if __name__ == "__main__":
    main()
