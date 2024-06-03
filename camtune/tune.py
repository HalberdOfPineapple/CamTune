import os
import sys
import yaml
import time
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
from camtune.database import BaseDatabase, DBMS_SET
from camtune.search_space import SearchSpace
from camtune.optimizer.optim_utils import tensor_to_config, get_default
from camtune.utils import (init_logger, print_log, get_log_dir, get_result_dir, set_expr_paths,
                           CONFIG_DIR, KNOB_DIR, OLAP_BENCHMARKS, TR_KNOB_DIR, DEVICE, DTYPE)
from camtune.utils.logger import log_results, log_results_explain



class ObjFuncWrapper:
    # db.step() takes a Configuration object as input
    # while the Tuner will suggest tensors as outputs to be evaluated
    # => need to convert tensors to Configuration objects
    # Note that perf and worst perf does not need to consider sign change before returning the result
    def __init__(
            self, 
            obj_func: callable, 
            search_space: SearchSpace, 
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
        self.worst_perf = 0 if not negate else -sys.float_info.max
        self.dummy_exec = dummy_exec

        self.result_dict = {}

    def __call__(self, sample_tensor: torch.Tensor) -> float:
        config: Configuration = None
        if sample_tensor is None:
            print_log('=' * 80, print_msg=True)
            print_log(f'[Tune] [Default Execution]', print_msg=True)
        else:
            config = tensor_to_config(sample_tensor, self.search_space)

            print_log('=' * 80, print_msg=True)
            print_log(f'[Tune] [Iteration {self.eval_counter}]', print_msg=True)
            for k, v in dict(config).items():
                print_log(f"\t{k}: {v}", print_msg=False)

        eval_result: dict = self.obj_func(config, dummy=self.dummy_exec)
        if not eval_result.get('exec_success', False):
            print_log(f"[Tune] [Iteration {self.eval_counter}]: Execution failed.")
            record_perf = 0 if not self.negate else -12000 #  self.worst_perf / 4 if not self.negate else self.worst_perf * 4
            eval_result[self.perf_name] = record_perf if not self.negate else -record_perf
            perf = eval_result[self.perf_name]
        else:
            perf = eval_result[self.perf_name]
            record_perf = perf if not self.negate else -perf

            # if negate, the optimizer should maximize the negative value to reduce the perf metric (e.g. latency)
            # if not negate, the optimizer should maximize the positive value to increase the perf metric (e.g. throughput)
            # => in both cases, the optimizer are doing the job of maximization and the worst perf should be the minimum value of the legal range 
            # (but in LlamaTune setting, the optimizer is by default minimizing the objective function valu)
            self.worst_perf = min(self.worst_perf, record_perf) if self.eval_counter > 0 else record_perf

        print_log(
                f'[Tune] [Iteration {self.eval_counter}]: {self.perf_name}: {perf:.3f} ({self.perf_unit})', print_msg=True)
        self.eval_counter += 1
        self.result_dict[str(self.eval_counter)] = {**eval_result, 'config': dict(config)}

        return record_perf

    def forward_config(self, config: Configuration) -> float:
        print_log('=' * 80, print_msg=True)
        print_log(f'[Tune] [Iteration {self.eval_counter}]', print_msg=True)
        for k, v in dict(config).items():
            print_log(f"\t{k}: {v}", print_msg=False)

        eval_result: dict = self.obj_func(config, dummy=self.dummy_exec)
        if not eval_result.get('exec_success', False):
            print_log(f"[Tune] [Iteration {self.eval_counter}]: Execution failed.")
            record_perf = 0 if not self.negate else -12000 #  self.worst_perf / 4 if not self.negate else self.worst_perf * 4
            eval_result[self.perf_name] = record_perf if not self.negate else -record_perf
            perf = eval_result[self.perf_name]
        else:
            perf = eval_result[self.perf_name]
            record_perf = perf if not self.negate else -perf

            # if negate, the optimizer should maximize the negative value to reduce the perf metric (e.g. latency)
            # if not negate, the optimizer should maximize the positive value to increase the perf metric (e.g. throughput)
            # => in both cases, the optimizer are doing the job of maximization and the worst perf should be the minimum value of the legal range 
            # (but in LlamaTune setting, the optimizer is by default minimizing the objective function valu)
            self.worst_perf = min(self.worst_perf, record_perf) if self.eval_counter > 0 else record_perf

        print_log(
                f'[Tune] [Iteration {self.eval_counter}]: {self.perf_name}: {perf:.3f} ({self.perf_unit})', print_msg=True)
        self.eval_counter += 1
        self.result_dict[str(self.eval_counter)] = {**eval_result, 'config': dict(config)}

        return record_perf

def save_config_data(
        result_X: torch.Tensor, 
        result_Y: torch.Tensor, 
        result_file_path: str, 
        search_space: SearchSpace,
        metric: str,
    ):
    with open(result_file_path, 'w') as f:
        result_dicts = {}
        for i, perf_val in enumerate(result_Y):
            config_tensor: torch.Tensor = result_X[i] # (D, )
            metric_val = float(perf_val[0])

            result_dict = {metric: metric_val}
            result_dict['config'] = dict(tensor_to_config(config_tensor, search_space))
            result_dicts[i] = result_dict

        json.dump(result_dicts, f)

def load_tr_range(dimension: int, benchmark_name: str, tr_range_name: str, search_space: SearchSpace) -> torch.Tensor:
    tr_range_path = os.path.join(TR_KNOB_DIR, benchmark_name, f'{tr_range_name}_tr_knobs.json')
    with open(tr_range_path, 'r') as f:
        tr_range_dict = json.load(f)
    
    tr_bounds = torch.zeros((2, dimension)).to(device=DEVICE, dtype=DTYPE)
    for knob_name, knob_range in tr_range_dict.items():
        knob_idx = search_space.knob_to_idx[knob_name]
        tr_bounds[0, knob_idx] = knob_range[0]
        tr_bounds[1, knob_idx] = knob_range[1]
    return tr_bounds
    

def main(expr_name: str, expr_config: dict, benchmark_name: str, dummy_exec: bool = False, on_wsl: bool = False, reboot: bool = False):
    init_start_time = perf_counter()

    # -----------------------------------
    # Setup logger
    set_expr_paths(expr_name, benchmark_name, dummy_exec=dummy_exec, algo_optim=False)
    log_idx, log_filename = init_logger(expr_name=expr_name)
    if expr_config['database'].get('exec_mode', 'raw') == 'explain':
        knob_definition_fn = expr_config['database']['knob_definitions']
        knob_definition_fn = knob_definition_fn.split('.')[0] + '_no_cost.json'
        expr_config['database']['knob_definitions'] = knob_definition_fn
        print_log(f"[Tune] [Explain Mode] Knob definition file: {knob_definition_fn}")

    is_test: bool = expr_config['tune']['num_evals'] <= 5
    
    result_file_name = f"{expr_name}_data.json" if log_idx == 0 else f'{expr_name}_{log_idx}_data.json'
    result_file_path = os.path.join(get_result_dir(), result_file_name)
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    data_file_name = f"{expr_name}{'' if log_idx == 0 else f'_{log_idx}'}_data.log"
    data_file_name = os.path.join(get_result_dir(), data_file_name)

    print_log('=' * 80, print_msg=True)
    print_log(f"[Tune] [Hosting Machine]: {socket.gethostname()}", print_msg=True)
    print_log(f"[Tune] [Dummy Mode]: {dummy_exec}", print_msg=True)
    print_log(f"[Tune] [Test Mode]: {is_test}", print_msg=True)
    print_log(f"[Tune] [On WSL]: {on_wsl}", print_msg=True)

    print_log(f"[Tune] [Log file path]: {log_filename}", print_msg=True)
    print_log(f"[Tune] [Result file path]: {result_file_path}", print_msg=True)
    print_log(f"[Tune] [Data file path]: {data_file_name}", print_msg=True)
    

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

    # -----------------------------------
    # Build Database Controller
    db_config: dict = expr_config['database']
    olap: bool = benchmark_name.split('_')[0].lower() in OLAP_BENCHMARKS

    # db = PostgresqlDB(db_config, on_wsl=on_wsl)
    db: BaseDatabase = DBMS_SET[db_config['db_type']](db_config, on_wsl=on_wsl)
    

    exec_mode: str = 'raw' if 'exec_mode' not in db_config else db_config['exec_mode']
    perf_name: str = db_config['perf_name']
    perf_unit: str = db_config['perf_unit']
    negate: bool = db_config['negate']
    clear_db_data: bool = not dummy_exec and db_config.get('vacuum_full', True) and not is_test

    # -----------------------------------
    # Setup search space from knob definition file
    knob_definition_path: str = os.path.join(
        KNOB_DIR, db_config['knob_definitions'])
    search_space = SearchSpace(
        knob_definition_path, 
        is_kv_config=True,
        seed=expr_config['tune']['seed'],
    )
    bounds: torch.Tensor = search_space.bounds
    discrete_dims: List[int] = search_space.discrete_dims
    cat_dims: List[int] = search_space.cat_dims
    integer_dims: List[int] = search_space.integer_dims
    default_conf_tensor = search_space.get_default_conf_tensor() # (1, dim)

    tr_range_name: Optional[str] = expr_config['tune'].get('tr_range_name', None)
    tr_bounds = load_tr_range(bounds.shape[-1], benchmark_name, tr_range_name, search_space) if tr_range_name is not None else None

    # -----------------------------------
    # Setup objective function
    obj_func = ObjFuncWrapper(
        db.step, search_space, negate,
        perf_name, perf_unit, 
        dummy_exec=dummy_exec
    )

    # ----------------------------------- 
    # Initialize tuner and start tuning
    tuner = Tuner(
        expr_name = expr_name, 
        args = expr_config['tune'], 
        obj_func = obj_func,
        bounds = bounds, # (2, D)
        discrete_dims = discrete_dims,
        cat_dims = cat_dims,
        integer_dims = integer_dims,
        default_conf=default_conf_tensor,
        definition_path=knob_definition_path,
        search_space=search_space,
        tr_bounds=tr_bounds,
    )

    #  --------------------------------------
    # Optimization
    init_time = None
    try:
        db.clear_sys_states()
        db.default_restart(exec=(not olap and not is_test), dummy=dummy_exec)
        init_time = perf_counter() - init_start_time

        start_time = perf_counter()
        try:
            result_X, result_Y = tuner.tune()
        except Exception as e:
            print_log(f"[Tune] Error in tuning: {e} with information: {traceback.format_exc()}")
            result_X, result_Y = tuner.optimizer.X, tuner.optimizer.Y
        tuning_time = perf_counter() - start_time

        # --------------------------------------------------------
        # Save and Logging results
        if len(result_X) > 0:
            best_config = tensor_to_config(result_X[result_Y.argmin()], search_space)
            best_Y = result_Y.max().item() if not negate else -result_Y.max().item()
            result_X: torch.Tensor = result_X.detach().cpu()
            result_Y: np.array = result_Y.detach().cpu().numpy() if not negate else -result_Y.detach().cpu().numpy()
            if exec_mode == 'explain':
                log_results_explain(best_config, db.step, db_config, tuning_time)
            else:
                log_results(best_config, best_Y, tuning_time, perf_name)
        else:
            print_log("[Tune] No valid result data (result_X) found.", print_msg=True)
        

        # --------------------------------------
        # Saving original data
        if hasattr(tuner.optimizer, 'get_original_data'):
            print_log("[Tune] Saving original data (within normalized space)...", print_msg=True)
            result_X, result_Y = tuner.optimizer.get_original_data()
            result_X, result_Y = result_X.detach().cpu().numpy(), result_Y.detach().cpu().numpy()
            with open(data_file_name, 'w') as f:
                for x, fx in zip(result_X, result_Y):
                    x = str(list(x))
                    f.write(f"{fx}, {x}\n")
    # --------------------------------------------------------
    # Clean up
    # --------------------------------------------------------
    except Exception as e:
        print_log(f"[Tune] Error in main: {e}\t{traceback.format_exc()}")
        raise e
    finally:
        with open(result_file_path, 'w') as f:
            json.dump(obj_func.result_dict, f)

        clean_start_time = perf_counter()
        clean_up(db, clear_db_data, is_test, dummy_exec, reboot)
        clean_time = perf_counter() - clean_start_time

        if init_time is not None:
            print_log(f"[Tune] Initialization time: {init_time:.3f} sec", print_msg=True)
        print_log(f"[Tune] Clean up time: {clean_time:.3f} sec", print_msg=True)

def clean_up(db: BaseDatabase, clear_db_data: bool, is_test: bool, dummy_exec: bool, reboot: bool):
    print_log(f"[Tune] Restarting the database with default configuration...", print_msg=True)
    restarted = True  
    try:
        db.default_restart(exec=False, dummy=dummy_exec)
        time.sleep(5)
    except Exception as e:
        print_log(f"[Tune] Error in restarting the database: {e}\t{traceback.format_exc()}", print_msg=True)
        restarted = False

    db.clear_sys_states()
    if restarted:
        db.clear_db_states(clear_db_data=clear_db_data)

    if not dummy_exec and not is_test and reboot:
        print_log("[Tune] Rebooting the remote machine...", print_msg=True)
        db.reboot()

def build_expr_config(benchmark_name, expr_name, num_evals: int, machine_name: str=None):
    db_config_path = os.path.join(CONFIG_DIR, 'benchmarks', f'{benchmark_name}.yaml')
    with open(db_config_path, 'r') as f:
        db_config: dict = yaml.safe_load(f)
    
    optim_config_path = os.path.join(CONFIG_DIR, 'optimizers', f'{expr_name}.yaml')
    with open(optim_config_path, 'r') as f:
        optim_config = yaml.safe_load(f)
    if num_evals == 50:
        if 'llama' in optim_config['optimizer']:
            optim_param = optim_config.get('optimizer_params', {})
            optim_param = optim_param.get('optimizer', {})
            optim_param['init_rand_samples'] = 15
        elif 'winter' in optim_config['optimizer']:
            optim_param = optim_config.get('optimizer_params', {})
            optim_param['global_num_init'] = 15
            optim_param['local_num_init'] = 10
        else:
            optim_param = optim_config.get('optimizer_params', {})
            optim_param['num_init'] = 15
        optim_config['optimizer_params'] = optim_param
    
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
    expr_config['tune']['num_evals'] = num_evals
    return expr_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr_name', '-e', type=str, help='Experiment configuration file name')
    parser.add_argument('--benchmark', '-b', type=str, default='tpch', help='Benchmark name')
    parser.add_argument('--machine', '-m', type=str, default=None, help='Machine and SSH config name')
    parser.add_argument('--dummy', '-d', action='store_true', help='Run dummy mode')
    parser.add_argument('--on_wsl', '-w', action='store_true', help='Run on WSL')
    parser.add_argument('--reboot', '-r', type=int, help='Reboot after experiment')
    parser.add_argument('--num_evals', '-n', type=int, default=100)
    parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed')
    parser.add_argument('--test', '-t', action='store_true', help='Run test mode')
    parser.add_argument('--tr_ranges', '-tr', type=str, default=None, help='Trust region name')
    parser.add_argument('--db_type', '-db', type=str, default='postgresql', help='Database type')
    args = parser.parse_args()
    
    dummy_exec: bool = args.dummy
    on_wsl: bool = args.on_wsl
    expr_name: str = args.expr_name
    benchmark_name: str = args.benchmark
    machine_name: Optional[str] = args.machine
    num_evals: int = args.num_evals
    reboot: bool = args.reboot == 1
    seed: int = args.seed
    is_test: bool = args.test
    tr_range_name: Optional[str] = args.tr_ranges
    dbms_type: str = args.db_type

    
    expr_config = build_expr_config(benchmark_name, expr_name, num_evals, machine_name)
    if not expr_name.endswith(f'_{num_evals}'):
        expr_name = f"{expr_name}_{num_evals}"
    if seed is not None:
        expr_config['tune']['seed'] = expr_config['tune'].get('seed', 0)
    if tr_range_name is not None:
        expr_config['tune']['tr_range_name'] = tr_range_name
    if dbms_type.lower() not in DBMS_SET:
        raise ValueError(f"[Tune] DBMS type {dbms_type} is not supported.")
    expr_config['database']['db_type'] = expr_config['database'].get('db_type', dbms_type)

    try:
        main(expr_name, expr_config, benchmark_name, dummy_exec, on_wsl, reboot)
    except Exception as e:
        print_log(f"[Tune] Tuning terminated because: {e}\t{traceback.format_exc()}", print_msg=True)
        exit(1)
