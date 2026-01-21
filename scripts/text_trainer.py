#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import json
import os
import shutil
import copy
import subprocess
import sys
import time 
from datetime import datetime, timezone, timedelta
import shlex
from typing import Optional, Sequence, Union

from state_manager import get_state, set_state
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib
from transformers import AutoConfig
import lr_utils

Command = Union[str, Sequence[str]]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_cmd_with_log(
    cmd: Command,
    log_file_path: str,
    env_vars: Optional[dict] = None,
    cwd: Optional[str] = None,
    check: bool = False,
) -> int:
    """
    Run a command, streaming stdout+stderr to both console and a log file.
    Returns the process return code. If check=True, raises RuntimeError on non-zero.
    """
    _ensure_parent_dir(log_file_path)
    with open(log_file_path, "w") as log_file:
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        use_shell = isinstance(cmd, str)
        process = subprocess.Popen(
            cmd,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
            cwd=cwd,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        log_file.write(f"\nProcess completed with return code: {return_code}\n")
        log_file.flush()

    if check and return_code != 0:
        raise RuntimeError(
            f"Command failed with return code {return_code}. See log: {log_file_path}"
        )
    return return_code


def _split_cmd(cmd: str) -> list[str]:
    # Be forgiving with quoted arguments. On Windows, shlex uses different rules.
    return shlex.split(cmd, posix=(os.name != "nt"))


def _join_cmd(tokens: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(tokens))
    return " ".join(shlex.quote(t) for t in tokens)


def get_arg_value(cmd: str, arg_name: str) -> Optional[str]:
    """Extract value for a CLI arg like '--learning_rate 1e-4' from a command string."""
    tokens = _split_cmd(cmd)
    flag = f"--{arg_name}"
    flag_eq = f"{flag}="
    value: Optional[str] = None
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            value = tokens[i + 1]
        elif t.startswith(flag_eq):
            value = t[len(flag_eq) :]
    return value


def set_arg_value(cmd: str, arg_name: str, arg_value: str) -> str:
    """
    Replace or append a CLI arg like '--output_dir <value>' in a command string.
    Preserves other args and supports quoted paths.
    """
    tokens = _split_cmd(cmd)
    flag = f"--{arg_name}"
    flag_eq = f"{flag}="
    for i, t in enumerate(tokens):
        if t == flag:
            if i + 1 < len(tokens):
                tokens[i + 1] = str(arg_value)
            else:
                tokens.append(str(arg_value))
            return _join_cmd(tokens)
        if t.startswith(flag_eq):
            tokens[i] = f"{flag_eq}{arg_value}"
            return _join_cmd(tokens)
    tokens.extend([flag, str(arg_value)])
    return _join_cmd(tokens)


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str) -> str:
    """Backwards-compatible wrapper used throughout this file."""
    return set_arg_value(cmd, arg_name, arg_value)


def extract_value_from_cmd(cmd: str, arg_name: str) -> Optional[str]:
    """Backwards-compatible wrapper used throughout this file."""
    return get_arg_value(cmd, arg_name)


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
OOM_ERROR_2 = "RuntimeError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif OOM_ERROR_2 in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _preserve_effective_batch(cmd: str, old_bs: int, new_bs: int, max_accum: int = 16) -> str:
    """
    If we reduce per-device batch size to avoid OOM, try to increase gradient_accumulation_steps
    to preserve effective batch size (better optimization stability).
    """
    if old_bs <= 0 or new_bs <= 0 or new_bs >= old_bs:
        return cmd
    old_accum = _as_int(extract_value_from_cmd(cmd, "gradient_accumulation_steps")) or 1
    target = old_bs * old_accum
    # ceil(target / new_bs)
    new_accum = int((target + new_bs - 1) // new_bs)
    new_accum = max(1, min(max_accum, new_accum))
    if new_accum != old_accum:
        print(
            f"Adjusting gradient_accumulation_steps from {old_accum} to {new_accum} to preserve effective batch",
            flush=True,
        )
        cmd = replace_args_in_cmd(cmd, "gradient_accumulation_steps", str(new_accum))
    return cmd


def _try_enable_gradient_checkpointing(cmd: str) -> str:
    gc = extract_value_from_cmd(cmd, "gradient_checkpointing")
    if gc is None:
        return cmd
    if str(gc).lower() in ("false", "0", "no"):
        print("Enabling gradient_checkpointing=True to reduce memory usage", flush=True)
        return replace_args_in_cmd(cmd, "gradient_checkpointing", "True")
    return cmd


def _try_reduce_vllm_util(cmd: str, floor: float = 0.25, step: float = 0.05) -> str:
    cur = extract_value_from_cmd(cmd, "vllm_gpu_memory_utilization")
    try:
        cur_f = float(cur) if cur is not None else None
    except Exception:
        cur_f = None
    if cur_f is None:
        return cmd
    new_f = max(floor, cur_f - step)
    if new_f < cur_f:
        print(f"Reducing vllm_gpu_memory_utilization from {cur_f} to {new_f}", flush=True)
        return replace_args_in_cmd(cmd, "vllm_gpu_memory_utilization", str(new_f))
    return cmd


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
):
    _ensure_parent_dir(log_path)
    for i in range(retries):
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size_i = _as_int(current_batch_size)
                    if current_batch_size_i is not None:
                        if current_batch_size_i > 1:
                            new_batch_size = max(1, current_batch_size_i // 2)
                            print(
                                f"Reducing batch size from {current_batch_size_i} to {new_batch_size}",
                                flush=True,
                            )
                            train_cmd = replace_args_in_cmd(
                                train_cmd,
                                "per_device_train_batch_size",
                                str(new_batch_size),
                            )
                            train_cmd = _preserve_effective_batch(
                                train_cmd, current_batch_size_i, new_batch_size
                            )
                            train_cmd = _try_enable_gradient_checkpointing(train_cmd)
                        else:
                            print("batch size is 1, cannot reduce further", flush=True)
                            if task_type == TaskType.GRPOTASK.value:
                                train_cmd = replace_args_in_cmd(
                                    train_cmd, "use_vllm", "False"
                                )
                            train_cmd = _try_enable_gradient_checkpointing(train_cmd)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        # First try reducing vLLM cache pressure; if it still fails on next retry, we'll disable vLLM.
                        train_cmd = _try_reduce_vllm_util(train_cmd)
                        print(f"VLLM OOM error, disable VLLM on next fallback if needed", flush=True)
                        # If utilization is already near floor, disable vLLM now.
                        use_vllm = extract_value_from_cmd(train_cmd, "use_vllm")
                        if use_vllm is None or str(use_vllm).lower() not in ("false", "0", "no"):
                            util = extract_value_from_cmd(train_cmd, "vllm_gpu_memory_utilization")
                            try:
                                util_f = float(util) if util is not None else 1.0
                            except Exception:
                                util_f = 1.0
                            if util_f <= 0.26:
                                print("vLLM utilization already low; disabling vLLM", flush=True)
                                train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        training_env_vars = {
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
            # Helps reduce allocator fragmentation-related OOMs.
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
        }

        rc = run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        if rc != 0:
            time.sleep(5)
            continue
        # check if training is successfully here so we can break the loop; if output_dir contains file: "successs.txt" return true
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if output_dir and os.path.exists(os.path.join(output_dir, "success.txt")):
            return True
        time.sleep(5)
    return False


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def delete_poor_checkpoints(train_runs: list[dict]):
    if not train_runs:
        return
    losses = [run.get("current_loss") for run in train_runs if run.get("current_loss") is not None]
    if not losses:
        return
    lowest_loss = min(losses)
    for run in train_runs:
        run_loss = run.get("current_loss")
        if run_loss is None:
            continue
        if run_loss > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.18,
        TaskType.DPOTASK.value: 0.18,
        TaskType.GRPOTASK.value: 0.2,
        TaskType.CHATTASK.value: 0.18,
    }
    return log_scale_map[task_type]


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=5)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.24383
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        # upgrade deepspeed
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        # install kernel
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "checking_mode": "first_time",
    }

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    run_cmd_with_log(
        tokenize_cmd,
        os.path.join(ds_folder, f"tokenize_{args.task_id}.log"),
        check=True,
    )

    original_train_cmd = train_cmd
    train_success = False
    state = {"mode": "initial"}
    set_state(state)  # reset first
    count = 0
    while True:
        state = get_state()
        train_cmd = original_train_cmd  # will replace based on the state later
        c_train_info = copy.deepcopy(train_info)
        final_output_dir = None
        resume_from_checkpoint = None
        if args.task_type == TaskType.GRPOTASK.value:
            # GRPO: run exactly once (no LR sweep / checking loop)
            state["mode"] = "finish"
            c_train_info["train_request"]["checking_mode"] = "none"
        else:
            if state["mode"] == "initial":
                c_train_info["train_request"]["checking_mode"] = "first_time"
                
            elif state["mode"] == "continue":
                c_train_info["train_request"]["checking_mode"] = "second_time"
                n_runs = int(state.get("next_runs", 0))
                if n_runs <= 0:
                    raise RuntimeError("State is 'continue' but missing/invalid 'next_runs'.")
                if "lrs" not in state: # first time of continue
                    current_lr = float(state["train"]["lr"])
                    state["lrs"] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=get_log_scale(args.task_type))
                    assert len(state["lrs"]) == n_runs, f"Number of learning rates {state['lrs']} should be equal to number of runs {n_runs}"
                    state["runs"] = []
                
                set_state(state)
                state["runs"].append(state["train"].copy())
                delete_poor_checkpoints(state["runs"])
                if len(state["runs"]) < n_runs:
                    index = len(state["runs"])
                    current_lr = state["lrs"][index]
                    train_cmd = replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                else: # the final run
                    # first find from runs the best loss
                    c_train_info["train_request"]["checking_mode"] = "none"
                    # Be robust to missing loss signals.
                    def _loss_key(run: dict) -> float:
                        v = run.get("current_loss")
                        return float(v) if v is not None else float("inf")
                    best_run = min(state["runs"], key=_loss_key)
                    index = state["runs"].index(best_run)
                    print(f"BL;{index};{best_run.get('current_loss')}; {state['lrs'][index]}", flush=True)
                    train_cmd = best_run["train_cmd"]
                    final_output_dir = best_run["output_dir"]
                    # Resume from the warmup checkpoint to avoid redoing the first checking_step steps.
                    resume_from_checkpoint = best_run.get("checkpoint_dir", None)
                    state["mode"] = "finish"
            else: # the state = finish; no need to run more
                assert state["mode"] == "finish"
                break
        
        set_state(state)
        if train_cmd:
            run_output_dir = output_dir + f"_{count}" if not final_output_dir else final_output_dir
            train_cmd = replace_args_in_cmd(train_cmd, "output_dir", run_output_dir)
            
            current_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_{count}.json")
            if resume_from_checkpoint:
                c_train_info["train_request"]["resume_from_checkpoint"] = resume_from_checkpoint
            with open(current_request_path, "w") as f:
                json.dump(c_train_info, f, indent=4, ensure_ascii=False)
            
            train_cmd = replace_args_in_cmd(train_cmd, "request_path", current_request_path)
            
            state["train"] = {
                "train_cmd": train_cmd,
                "log_path": os.path.join(ds_folder, f"train_{args.task_id}.log"),
                "lr": extract_value_from_cmd(train_cmd, "learning_rate"),
                "output_dir": run_output_dir
            }
            state["train"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            set_state(state)
            
            log_path = state["train"]["log_path"]
            # print(f"Run training with train_info: {c_train_info}", flush=True)
            success = run_training(
                train_cmd,
                log_path,
                args.task_id,
                args.retries,
                args.task_type,
                args.expected_repo_name,
            )
            time.sleep(5)
            if not success:
                print(f"Training failed for task {args.task_id} at count={count}", flush=True)
                break 
        
        count += 1
        if args.task_type == TaskType.GRPOTASK.value:
            break

    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f"Training failed for task {args.task_id}", flush=True)
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
