import os
import sys
import yaml
import json
import torch
import socket
import argparse
import traceback
import numpy as np
from time import perf_counter
from typing import List, Optional
from ConfigSpace import ConfigurationSpace, Configuration

from camtune.tuner import Tuner
from camtune.database import PostgresqlDB
from camtune.search_space import SearchSpace
from camtune.optimizer.optim_utils import tensor_to_config, get_default
from camtune.utils import (get_result_dir,
                           CONFIG_DIR, KNOB_DIR, OLAP_BENCHMARKS)
from camtune.utils.logger import log_results, log_results_explain



def main_from_file(expr_name: str, expr_config: dict, benchmark_name: str, expr_idx: int, knob_idx: int):
    # -----------------------------------
    result_file_name = f"{expr_name}_data.json" if expr_idx == 0 else f'{expr_name}_{expr_idx}_data.json'
    result_file_path = os.path.join(get_result_dir(), benchmark_name, result_file_name)
    with open(result_file_path, 'r') as f:
        results = json.load(f)
    target_knob_dict = results[knob_idx]['config']

    # -----------------------------------
    # Logging configurations
    print("=" * 50, print_msg=True)
    print(f"[Knob Evaluation] {knob_idx}-th knob from {result_file_path}", print_msg=True)
    for key, value in target_knob_dict.items():
        print(f"{key}: {value}")

    # -----------------------------------
    # Build Database Controller
    db_config: dict = expr_config['database']
    db = PostgresqlDB(db_config)
    db.default_restart(exec=False)

    db.knob_applier.apply_knobs(target_knob_dict, online=False)

def main_single(expr_name: str, db_config: dict, knob_file_path: str):
    db = PostgresqlDB(db_config)
    db.executor.recover_default_config()

    with open(knob_file_path, 'r') as f:
        knob_dict = json.load(f)
    print(f"[Knob Evaluation] Evaluating knobs from {knob_file_path}")
    for key, value in knob_dict.items():
        print(f"\t{key}:\t{value} (type={type(value)})")

    knob_applier = db.knob_applier
    success = knob_applier.kill_postgres()
    if not success:
        raise RuntimeError("[PGKnobApplier] PostgreSQL failed to shut down before applying knobs offline.")

    knobs_not_in_cnf = knob_applier.modify_config_file(knob_dict)
    db.knob_applier.start_postgres()
    # or manualy execute: echo CL741286% | sudo -S -u postgres /usr/lib/postgresql/16/bin/postgres --config_file=/home/wl446/pg_data/postgresql/16/tune_conf.conf -D /home/wl446/pg_data/postgresql/16/main


def build_expr_config(benchmark_name, expr_name, machine_name: str=None):
    db_config_path = os.path.join(CONFIG_DIR, 'benchmarks', f'{benchmark_name}.yaml')
    with open(db_config_path, 'r') as f:
        db_config: dict = yaml.safe_load(f)
    
    optim_config_path = os.path.join(CONFIG_DIR, 'optimizers', f'{expr_name}.yaml')
    with open(optim_config_path, 'r') as f:
        optim_config = yaml.safe_load(f)
    
    if machine_name:
        ssh_config_path = os.path.join(CONFIG_DIR, 'machine', f'{machine_name}.yaml')
        with open(ssh_config_path, 'r') as f:
            ssh_config: dict = yaml.safe_load(f)

        # concatenate machine config to db config
        db_config.update(ssh_config)
        if ssh_config.get('db_host', 'localhost') != 'localhost':
            db_config['remote_mode'] = True

    expr_config = {
        'database': db_config,
        'tune': optim_config
    }
    return expr_config

def log_to_dict(log_path: str, target_json_path: str=None):
    with open(log_path, 'r') as f:
        raw_log_str = f.read()
    log_lines = raw_log_str.strip().split('\n')

    res_dict = {}
    for line in log_lines:
        # Remove the logging prefix
        stripped_line = line.split('-')[-1].strip()
        
        # Split the line into key and value
        if ':' in stripped_line:
            key, value = stripped_line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Check if the value is numeric
            if value.replace('.', '', 1).isdigit():
                # It's a number, float or int
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            
            res_dict[key] = value

    if target_json_path is not None:
        with open(target_json_path, 'w') as f:
            json.dump(res_dict, f, indent=4)
    return res_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', '-M', type=str, default='single', help='Evaluation mode (single or from result file)')
    parser.add_argument('--expr_name', '-e', type=str, help='Experiment configuration file name')
    parser.add_argument('--benchmark', '-b', type=str, default='tpch', help='Benchmark name')
    parser.add_argument('--machine', '-m', type=str, default=None, help='Machine and SSH config name')
    parser.add_argument('--expr_idx', '-i', type=int, default=0, help='Experiment index')
    parser.add_argument('--knob_idx', '-k', type=int, default=0, help='Index of selected knob in the result file')
    parser.add_argument('--log_file_name', '-l', type=str, default=None, help='Path to the log file')
    parser.add_argument('--knob_file_name', '-f', type=str, default=None, help='Path to the knob file')
    parser.add_argument('--num_evals', '-n', type=int, default=100, help='Number of evaluations')
    args = parser.parse_args()
    
    expr_name = args.expr_name
    benchmark_name = args.benchmark
    machine_name = args.machine
    num_evals = args.num_evals
        
    expr_config = build_expr_config(benchmark_name, expr_name, machine_name)
    expr_config['tune']['num_evals'] = num_evals
    if not expr_name.endswith(f'_{num_evals}'):
        expr_name = f"{expr_name}_{num_evals}"

    if args.eval_mode == 'from_file':
        expr_idx = args.expr_idx
        knob_idx = args.knob_idx
        main_from_file(expr_name, expr_config, benchmark_name, expr_idx, knob_idx)
    
    elif args.eval_mode == 'single':
        log_file_name = args.log_file_name
        knob_file_name = args.knob_file_name

        knob_file_path = os.path.join(KNOB_DIR, 'expr_knobs', f'{knob_file_name}.json')
        if log_file_name is not None:
            log_file_path = os.path.join(KNOB_DIR, 'expr_knobs', f'{log_file_name}.txt')
            log_to_dict(log_file_path, knob_file_path)

        main_single(expr_name, expr_config['database'], knob_file_path)

