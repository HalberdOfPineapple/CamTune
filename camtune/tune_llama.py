import os
import sys
import yaml
import json
import torch
import argparse
import numpy as np
from time import perf_counter
from typing import List, Optional

from smac.facade.smac_ac_facade import SMAC4AC
from ConfigSpace import Configuration

from camtune.database import PostgresqlDB
from camtune.utils import (init_logger, print_log, set_expr_paths,
                           get_log_dir, get_result_dir, CONFIG_DIR, KNOB_DIR,
                           OLAP_BENCHMARKS, BENCHMARKS)
from camtune.utils.paths_llama import set_smac_output_dir
from camtune.utils.logger import log_results, log_results_explain
from camtune.optimizer.llama_utils.space import ConfigSpaceGenerator
from camtune.optimizer.llama_utils.get_smac import get_smac_optimizer


class ObjFuncLlama:
    def __init__(
            self, 
            obj_func: callable, 
            search_space: ConfigSpaceGenerator, 
            negate: bool, 
            perf_name: str, 
            perf_unit: str,
            dummy_exec: bool = False,
    ):
        self.obj_func = obj_func
        self.search_space = search_space
        self.negate = negate

        self.perf_name = perf_name
        self.perf_unit = perf_unit

        self.eval_counter = 0

        # if negate, the optimizer should minimize the objective function value to maximize the perf metric (e.g. throughput)
        # if positive, the optimizer should minimize the objective function value to reduce the perf metric (e.g. latency)
        # => in both cases, the optimizer are minimizing the objective function and the worst perf should always be the maximum in the legal range
        self.worst_perf = sys.float_info.max if not negate else 0
        self.dummy_exec = dummy_exec
    
    def __call__(self, config: Configuration) -> float:
        if config is None:
            print_log('=' * 80, print_msg=True)
            print_log(f'[Tune] [Default Execution]', print_msg=True)
            config = self.search_space.input_space.get_default_configuration()
        else:
            config = self.search_space.unproject_input_point(config)
            config = self.search_space.finalize_conf(config)
            print_log('=' * 80, print_msg=True)
            print_log(f'[Tune] [Iteration {self.eval_counter}]', print_msg=True)
            for k, v in dict(config).items():
                print_log(f"\t{k}: {v}", print_msg=False)

        eval_result: dict = self.obj_func(config, dummy=self.dummy_exec)
        if eval_result['exec_success'] == False:
            print_log(f"[Tune] [Iteration {self.eval_counter}]: Knob application failed.")
            self.eval_counter += 1
            return -self.worst_perf * 4 if self.negate else 0

        perf = eval_result[self.perf_name]
        record_perf = perf if not self.negate else -perf
        print_log(
            f'[Tune] [Iteration {self.eval_counter}]: {self.perf_name}: {perf:.3f} ({self.perf_unit})', print_msg=True)
        
        # Note that worst_perf should always be the maximum in the legal range
        self.worst_perf = max(self.worst_perf, record_perf) if self.eval_counter > 0 else record_perf

        self.eval_counter += 1
        return record_perf

def main(expr_name: str, expr_config: dict, benchmark_name: str, dummy_exec: bool = False, on_wsl: bool = False):
    # -----------------------------------
    # Setup logger
    set_expr_paths(expr_name, benchmark_name, dummy_exec=dummy_exec, algo_optim=False)
    expr_name = f"{expr_name}"
    log_idx = init_logger(expr_name=expr_name, benchmark_name=benchmark_name, log_dir=get_log_dir())

    # -----------------------------------
    # Logging configurations
    print_log("=" * 50, print_msg=True)
    print_log(f"Start Experiment: {expr_name}", print_msg=True)
    for k, v in expr_config.items():
        if 'pwd' in k or 'passwd' in k:
            continue

        if isinstance(v, dict):
            print_log(f"{k}:", print_msg=True)
            for kk, vv in v.items():
                print_log(f"\t{kk}: {vv}", print_msg=True)
        else:
            print_log(f"{k}: {v}", print_msg=True)
    print_log("=" * 50, print_msg=True)

    
    # -----------------------------------
    # Build Database Controller
    db = PostgresqlDB(expr_config, on_wsl=on_wsl)
    olap: bool = benchmark_name.lower() in OLAP_BENCHMARKS

    db.default_restart(exec=(not olap), dummy=dummy_exec)
    db_config: dict = expr_config['database']

    exec_mode: str = 'raw' if 'exec_mode' not in db_config else db_config['exec_mode']
    perf_name: str = db_config['perf_name']
    perf_unit: str = db_config['perf_unit']
    negate: bool = db_config['negate']
    vacuum: bool = not dummy_exec and not olap \
            and ('vacuum_full' not in db_config or db_config['vacuum_full'])

    if db_config['benchmark'].lower() == 'sysbench' and db_config['sysbench_prepare']:
        db.prepare_sysbench_data()

    # -----------------------------------
    # Setup search space from knob definition file
    llama_config = expr_config['llama']
    knob_definition_name: str =  db_config['knob_definitions'].split('.')[0]

    # Define search space
    # space = {'definition': 'postgres-9.6', 'ignore': 'postgres-none', 'adapter_alias': 'hesbo', 'le_low_dim': '8', 'target_metric': 'throughput'}
    space_dict = {
        'definition': knob_definition_name,
        'ignore': 'postgres-none',
        'adapter_alias': llama_config['adapter_alias'],
        'le_low_dim': llama_config['le_low_dim'],
        'target_metric': perf_name
    }
    search_space = ConfigSpaceGenerator.from_config(space_dict, seed=llama_config['seed'])

    # -----------------------------------
    # Setup objective function
    # db.step() takes a Configuration object as input
    # while the Tuner will suggest tensors as outputs to be evaluated
    # => need to convert tensors to Configuration objects
    # Note that perf and worst perf does not need to consider sign change before returning the result
    obj_func = ObjFuncLlama(
        db.step, search_space, negate,
        perf_name, perf_unit,
        dummy_exec=dummy_exec
    )

    # -----------------------------------
    # Initialize optimizer
    optimizer: SMAC4AC = get_smac_optimizer(
        expr_config, 
        search_space, 
        obj_func, 
    )

    # --------------------------------------
    # Optimization
    start_time = perf_counter()

    optimizer.optimize()

    elapsed_time = perf_counter() - start_time
    print_log(f"Elapsed time: {elapsed_time:.2f} seconds", print_msg=True)

    
    # --------------------------------------------------------
    # Setup saving path
    data_file_name = f"{expr_name}_data.json" if log_idx == 0 else f'{expr_name}_{log_idx}.json'
    result_file_path = os.path.join(get_result_dir(), benchmark_name, data_file_name)
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    # --------------------------------------------------------
    # Saving results
    metric = f"{perf_name} ({perf_unit})"
    run_history = optimizer.runhistory
    result_dicts = {}

    best_config: Configuration
    best_y: float = 0 if negate else sys.float_info.max
    for run_key, run_value in run_history.data.items():
        config_id = run_key.config_id

        config = dict(run_history.ids_config[config_id])
        config = search_space.unproject_input_point(config)
        config = search_space.finalize_conf(config)
        
        metric_val = float(run_value.cost)
        if metric_val < best_y:
            best_y = metric_val
            best_config = config
        metric_val = -metric_val if negate else metric_val

        result_dict = {metric: metric_val}
        result_dict['config'] = config
        result_dicts[config_id] = result_dict

    with open(result_file_path, 'w') as f:
        json.dump(result_dicts, f)
    
    # --------------------------------------------------------
    # Logging results
    if exec_mode == 'explain':
        log_results_explain(best_config, db.step, db_config, elapsed_time)
    else:
        log_results(best_config, best_y, elapsed_time, perf_name)

    # --------------------------------------------------------
    # Clean up
    # --------------------------------------------------------
    if db_config['benchmark'].lower() == 'sysbench' and db_config['sysbench_cleanup']:
        db.cleanup_sysbench_data()
    if vacuum: db.do_vaccum_full()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr_name', '-e', type=str, help='Experiment configuration file name')
    parser.add_argument('--benchmark', '-b', type=str, default='tpch', help='Benchmark name')
    parser.add_argument('--dummy', '-d', action='store_true', help='Run dummy mode')
    parser.add_argument('--on_wsl', '-w', action='store_true', help='Run on WSL')
    args = parser.parse_args()
    
    benchmark_name: str = args.benchmark
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f'Invalid benchmark name: {benchmark_name}')
    
    expr_name: str = args.expr_name
    config_file_path = os.path.join(CONFIG_DIR, benchmark_name, f'{expr_name}.yaml')
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'llama' not in config:
        raise ValueError(f'[TuneLlama] Experiment configuration does not contain llama configuration')

    dummy_exec: bool = args.dummy
    on_wsl: bool = args.on_wsl
    main(expr_name, config, benchmark_name, dummy_exec, on_wsl)
