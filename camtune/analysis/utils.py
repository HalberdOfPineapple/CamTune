import os 
import re
import yaml
import json
import torch
import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from typing import List, Dict, Optional, Callable, Tuple

from camtune.search_space import SearchSpace
from camtune.optimizer.optim_utils import unnormalize
from camtune.utils import (
    KNOB_DIR, BASE_DIR,
    OPTIM_CONFIG_DIR, OPTIM_LOG_DIR, OPTIM_RES_DIR,
    RESULT_DIR, LOG_DIR, CONFIG_DIR
)

algo_with_tr = {
    'turbo',
    'sturbo',
    'winter',
    # 'spring',
}

# --------------------------------------------------------------------------------
# Analysis Initialization
algo_optim = True
log_dir, result_dir, config_dir = None, None, None

def init_analysis_vars(use_algo_optim: bool, luchen: bool=False):
    global algo_optim, log_dir, result_dir, config_dir
    algo_optim = use_algo_optim

    if algo_optim:
        log_dir, result_dir, config_dir = OPTIM_LOG_DIR, OPTIM_RES_DIR, OPTIM_CONFIG_DIR
        if luchen:
            luchen_dir = os.path.join(BASE_DIR, 'optimizer', 'luchen_results')
            log_dir = os.path.join(luchen_dir, 'optim_logs')
            result_dir = os.path.join(luchen_dir, 'optim_results')
    else:
        log_dir, result_dir, config_dir = LOG_DIR, RESULT_DIR, CONFIG_DIR

    print('Initialized analysis file paths:')
    print(f'Log dir: {log_dir}')
    print(f'Result dir: {result_dir}')
    print(f'Config dir: {config_dir}')

    return config_dir, log_dir, result_dir

# --------------------------------------------------------------------------------
# Common Utilities

def convert_keys_to_int(data_dict: dict):
    """Recursively change all the key values that are digit strings to integers"""
    new_data_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            value = convert_keys_to_int(value)
        new_data_dict[int(key) if isinstance(key, str) and key.isdigit() else key] = value
    return new_data_dict

def configurations_to_dataframe(configurations, search_space):
    """
    Convert a list of Configuration objects to a pandas DataFrame.
    
    Args:
    - configurations (list of Configuration): The configurations to convert.
    
    Returns:
    - DataFrame: A DataFrame where each row is a configuration and each column is a hyperparameter.
    """
    # Extract hyperparameter names from the first configuration

    keys = list(dict(configurations[0]).keys())
    
    # Initialize a dictionary to hold data
    data = {key: [] for key in keys}
    
    # Populate the dictionary with data from each configuration
    for config in configurations:
        for key in keys:
            i = search_space.knob_to_idx[key]
            if i in search_space.enum_val2idx:
                data[key].append(search_space.enum_val2idx[i][config[key]])
            else:
                data[key].append(config[key])
    
    # Convert dictionary to DataFrame
    return pd.DataFrame(data)

algo_normalized = {
    'gp': True, 
    'winter': True,
    'turbo': True,
    'llama': False,
    'random': False,
    'sturbo': False,
    'bucket': False,
    'casmo': False,
}
def check_normalized(expr_name):
    for algo in algo_normalized.keys():
        if algo in expr_name and not (algo == 'turbo' and 'sturbo' in expr_name):
            return algo_normalized[algo]
    return False

def filter_by_ignore_list(algo: str, ignore_list: List[str]) -> bool:
    for ignore in ignore_list:
        if ignore in algo: 
            return True
    return False

def filter_by_include_list(algo: str, include_list: List[str]) -> bool:
    if len(include_list) == 0:
        return True

    for include in include_list:
        if include in algo: 
            return True
    return False

# --------------------------------------------------------------------------------
# Reading Data
def load_from_config(expr_name):
    # Load MCTS optimizer from configuration
    if algo_optim:
        config_file_name = os.path.join(config_dir, f'{expr_name}.yaml')
    else:
        config_file_name = os.path.join(config_dir, 'optimizers', f'{expr_name}.yaml')

    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def load_benchmark_config(benchmark_name):
    # Load benchmark configuration
    benchmark_config_path = os.path.join(config_dir, 'benchmarks', f'{benchmark_name}.yaml')
    with open(benchmark_config_path, 'r') as f:
        benchmark_config = yaml.safe_load(f)
    return benchmark_config

def get_time_from_log(benchmark_name, expr_name, idx=0):
    # sample line: Elapsed time: 5560.24 seconds
    log_file = get_expr_file(benchmark_name, expr_name, idx, file_type='log')
    print(f'Extracting time information from {log_file}...')

    time = None
    with open(log_file, 'r') as file:
        for line in file:
            if 'Elapsed time' in line:
                time = line.split('[Tune]')[-1].strip().split(':')[-1].strip().split(' ')[0]
                time = float(time)
                break
    return time
    
def get_tr_range_from_log(benchmark_name, expr_name, idx=0):
    # example line: 2024-05-27 16:32:22,935 - INFO - 	tr_range_name: turbo_cop_IV_temp_50
    log_file = get_expr_file(benchmark_name, expr_name, idx, file_type='log')
    print(f'Extracting TR range information from {log_file}...')

    tr_range = None
    with open(log_file, 'r') as file:
        for line in file:
            if 'tr_range_name' in line:
                tr_range = line.split('tr_range_name: ')[-1].strip()
                break
    return tr_range

def get_date_from_log(benchmark_name, expr_name, idx=0):
    log_file = get_expr_file(benchmark_name, expr_name, idx, file_type='log')
    print(f'Extracting date information from {log_file}...')

    date = None
    with open(log_file, 'r') as file:
        # example line: 2024-05-26 00:12:04,777 - INFO - 
        for line in file:
            if 'INFO' in line:
                date = line.split(' ')[0]
                break
    return date

def get_expr_file(benchmark_name, expr_name, idx=0, file_type: str='log'):
    dir = log_dir if file_type == 'log' else result_dir
    if file_type == 'log':
        file_name = f'{expr_name}.log' if idx == 0 else f'{expr_name}_{idx}.log'
    elif file_type == 'data':
        file_name = f'{expr_name}_data.log' if idx == 0 else f'{expr_name}_{idx}_data.log'
    elif file_type == 'tr':
        file_name = f'{expr_name}_tr.json' if idx == 0 else f'{expr_name}_{idx}_tr.json'
    else: # file_type == 'result'
        file_name = f'{expr_name}_data.json' if idx == 0 else f'{expr_name}_{idx}_data.json'
    return os.path.join(dir, benchmark_name, file_name)

def read_data(benchmark_name, expr_name, idx=0):
    data_path = get_expr_file(benchmark_name, expr_name, idx, file_type='data')
    print(f"Reading data from {data_path}...")

    function_values = []
    data_values = []
    with open(data_path, 'r') as file:
        for line in file:
            # Split the line into function value and coordinates, then extract the function value
            func_val = line.split(', [')[0].strip()
            func_val = func_val.replace('[', '').replace(']', '')
            function_values.append(float(func_val))

            # Extract the coordinates
            data = line.split(', [')[1].split(']')[0].strip().split(',')
            data = [float(d) for d in data]
            data_values.append(data)

    return np.array(function_values), np.array(data_values)

def read_result(benchmark_name, expr_name, perf_name=None, perf_unit=None, idx=0, include_all=False):
    result_path = get_expr_file(benchmark_name, expr_name, idx, file_type='result')
    print(f"Reading results from {result_path}...")

    with open(result_path, 'r') as file:
        results: dict = json.load(file)

    for iter, res_dict in results.items():
        if 'excution_time' in res_dict or 'execution_time' in res_dict:
            time_dict = res_dict['excution_time'] if 'excution_time' in res_dict else res_dict['execution_time']
            total_exec_time = sum([exec_time for exec_time in time_dict.values()])
            results[iter]['total_exec_time'] = total_exec_time
        else:
            results[iter]['total_exec_time'] = 12000
        
    if include_all:
        return results

    function_values, configs = [], []
    for i, res_dict in results.items():
        if perf_name == 'latency' and 'latency' not in res_dict: res_dict['latency'] = 100000
        function_values.append(res_dict[perf_name])
        configs.append(res_dict['config'])
    return np.array(function_values), np.array(configs)

def read_tr(benchmark_name, expr_name, expr_idx=0):
    tr_path = get_expr_file(benchmark_name, expr_name, expr_idx, file_type='tr')
    print(f"Reading TR data from {tr_path}...")

    with open(tr_path, 'r') as f:
        tr_data = json.load(f)
    tr_data = convert_keys_to_int(tr_data)
    return tr_data

def get_knob_dict(benchmark_config: dict):
    knob_dict_name = benchmark_config['knob_definitions']
    knob_dict_path = os.path.join(KNOB_DIR, f"{knob_dict_name}")
    with open(knob_dict_path, 'r') as f:
        knob_dict = yaml.safe_load(f)
    return knob_dict

def get_expr_configs(
    benchmark_name: str, 
    sample_expr_name: str, 
):
    benchmark_config_path = os.path.join(CONFIG_DIR, "benchmarks", f"{benchmark_name}.yaml")
    with open(benchmark_config_path, 'r') as f:
        benchmark_config = yaml.safe_load(f)
    knob_dict_name = benchmark_config['knob_definitions']
    knob_dict_path = os.path.join(KNOB_DIR, f"{knob_dict_name}")
    # negate = benchmark_config['negate']
    # metric = f"{benchmark_config['perf_name']}_{benchmark_config['perf_unit']}"

    optim_config_path = os.path.join(CONFIG_DIR, "optimizers", f"{'_'.join(sample_expr_name.split('_')[:-1])}.yaml")
    with open(optim_config_path, 'r') as f:
        optim_config = yaml.safe_load(f)
    seed = optim_config['seed']

    search_space = SearchSpace(
        knob_definition_path=knob_dict_path,
        is_kv_config=True,
        seed=seed,
    )
    # bounds: torch.Tensor = search_space.bounds.cpu()
    
    expr_names = [f.split('.')[0] for f in os.listdir(os.path.join(log_dir, benchmark_name)) if f.endswith('.log')]
    print(f'Detected experiment records: {expr_names}')
    return expr_names, benchmark_config, search_space
    
def data_from_config(
    expr_configs: List[Configuration],
    search_space: SearchSpace,
) -> np.array:
    expr_data = np.empty((len(expr_configs), len(search_space.input_variables)))
    for i, config in enumerate(expr_configs):
        for knob in dict(config).keys():
            if knob in search_space.knob_to_idx:
                idx = search_space.knob_to_idx[knob]
                if idx in search_space.enum_val2idx:
                    expr_data[i][idx] = search_space.enum_val2idx[idx][config[knob]]
                else:
                    expr_data[i][idx] = config[knob]  
    return expr_data

def split_expr_name(orig_expr_name: str) -> Tuple[str, int]:
    digit_split_name = [int(s) for s in orig_expr_name.split('_') if s.isdigit()]
    if len(digit_split_name) > 1:
        expr_idx = int(digit_split_name[-1])
        expr_name = '_'.join(orig_expr_name.split('_')[:-1])
    else:
        expr_idx = 0
        expr_name = orig_expr_name
    return expr_name, expr_idx

def split_expr_to_algo(expr_name: str) -> str:
    digit_split_name = [int(s) for s in expr_name.split('_') if s.isdigit()]
    if len(digit_split_name) > 1:
        expr_len = int(digit_split_name[-2])
        expr_idx = int(digit_split_name[-1])
        algo = '_'.join(expr_name.split('_')[:-1])
    else:
        expr_len = int(digit_split_name[-1])
        expr_idx = 0
        algo = expr_name
    return algo, expr_len, expr_idx


def read_expr_data(
    benchmark_name: str,
    benchmark_config: Dict[str, any],
    expr_names: List[str],
    search_space: SearchSpace,
    condition_func: Optional[Callable]=None,
    ignore_list: List[str] = [],
    include_list: List[str] = [],
    perf_name: Optional[str] = None,
    date_factor: Dict[str, float] = {},
):
    bounds: torch.Tensor = search_space.bounds.cpu()
    metric: str = benchmark_config['perf_name'] if perf_name is None else perf_name
    negate: bool = benchmark_config['negate']

    expr_name_list = []
    expr_val_list, expr_data_list, expr_config_list = [], [], []
    for orig_expr_name in expr_names:
        if filter_by_ignore_list(orig_expr_name, ignore_list) and \
            not (len(include_list) > 0 and filter_by_include_list(orig_expr_name, include_list)): 
            continue

        try:
            print('-' * 80)
            expr_name, expr_idx = split_expr_name(orig_expr_name)
            expr_vals, expr_configs = read_result(benchmark_name, expr_name, metric, idx=expr_idx)
            try:
                expr_vals, expr_data = read_data(benchmark_name, expr_name, expr_idx)
                if check_normalized(expr_name):
                    expr_data = unnormalize(torch.tensor(expr_data), bounds).numpy()
                    print(expr_data.shape)
                expr_vals = expr_vals if not negate else -expr_vals
            except Exception as e:
                print(f"Error reading data for {orig_expr_name} with error: {e} -> Read data from result configuration instead.")
                # Note that data from configuration are already unnormalized
                expr_data = data_from_config(expr_configs, search_space)
            
            expr_day = int(get_date_from_log(benchmark_name, expr_name, expr_idx).split('-')[-1])
            expr_vals = expr_vals * date_factor.get(expr_day, 1.0)

            print(f'Expr {orig_expr_name} has {len(expr_vals)} values, {len(expr_data)} data entries and {len(expr_configs)} configs')
        except Exception as e:
            print(f"Error reading {orig_expr_name} with error: {e}")
            continue
        min_len = min(len(expr_vals), len(expr_data), len(expr_configs))

        expr_val_list.append(expr_vals[:min_len])
        expr_data_list.append(expr_data[:min_len])
        expr_config_list.append(expr_configs[:min_len])
        for _ in range(min_len):
            expr_name_list.append(orig_expr_name)

    # concatentate the values and data
    expr_vals = np.concatenate(expr_val_list)
    expr_data = np.concatenate(expr_data_list)

    # Put all config list together
    expr_configs = []
    for expr_config in expr_config_list:
        for config in expr_config:
            if len(dict(config)) != len(search_space.input_variables):
                config_dict = {k: v for k, v in dict(config).items() if k in search_space.knob_to_idx}
                config = Configuration(search_space.input_space, config_dict)
            expr_configs.append(config)

    # ---------------------------------------------------------------
    print('-' * 80)
    condition_func = condition_func if condition_func is not None else lambda val, config: True
    expr_data_tups = [(val, data, config, name) for val, data, config, name in zip(expr_vals, expr_data, expr_configs, expr_name_list) if condition_func(val, config)]

    X = np.array([data for _, data, _, _ in expr_data_tups])
    y = np.array([val for val, _, _, _ in expr_data_tups])
    expr_configs = [config for _, _, config, _ in expr_data_tups]
    idx_to_expr_name = [name for _, _, _, name in expr_data_tups]

    config_df = configurations_to_dataframe(expr_configs, search_space)
    return X, y, expr_configs, idx_to_expr_name, config_df

    
# ------------------------------------------------------------------------------------------
# Extracting information from log file
def extract_split_idxes(benchmark_name, expr_name, idx=0):
    log_file =  get_expr_file(benchmark_name, expr_name, idx, file_type='log')
    print(f'Extracting split information from {log_file}...')

    # Regular expression to match lines with MCTS split information
    
    if 'turbo' in expr_name: 
        pattern = re.compile(r'\[TuRBO\] Iterations where TuRBO restarts: \[(.*)\]')
    elif 'winter' in expr_name:
        pattern = re.compile(r'\[WinterMCTS\] Iterations where search tree is rebuilt: \[(.*)\]')
    else:
        raise ValueError(f'Unknown optimizer: {expr_name} for split extraction.')

    idxes = []
    with open(log_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract the iteration numbers from the line
                idxes = [int(x) for x in re.findall(r'\d+', match.group(1))]
    print(f'Extracted split idxes: {idxes}')
    return idxes


def extract_mcts_expr_info(benchmark_name, expr_name, idx=0):
    log_file = get_expr_file(benchmark_name, expr_name, idx, log=True)
    print(log_file)

    expr_vals, expr_data = read_data(benchmark_name, expr_name, idx)
    optim_config: dict = load_from_config(expr_name)

    # Split data by the restart iterations
    num_evals = len(expr_vals)
    global_num_init = optim_config['optimizer_params']['global_num_init']
    local_num_init = optim_config['optimizer_params']['local_num_init']

    restart_iters = extract_split_idxes(benchmark_name, expr_name, idx)
    restart_data_dict = {}
    for i, restart_iter in enumerate(restart_iters):
        if restart_iter >= num_evals: break
        num_samples = restart_iters[i + 1] - restart_iter if i + 1 < len(restart_iters) else num_evals - restart_iter

        random_sample_data = expr_data[restart_iter: restart_iter + local_num_init]
        random_sample_vals = expr_vals[restart_iter: restart_iter + local_num_init]
        random_data = (random_sample_vals, random_sample_data)

        optim_sample_data = expr_data[restart_iter + local_num_init: restart_iter + num_samples]
        optim_sample_vals = expr_vals[restart_iter + local_num_init: restart_iter + num_samples]
        optim_data = (optim_sample_vals, optim_sample_data)

        num_samples = len(optim_sample_vals) + len(random_sample_vals)
        best_val = np.min(np.concatenate((optim_sample_vals, random_sample_vals), axis=0))
        best_val_idx = np.argmin(np.concatenate((optim_sample_vals, random_sample_vals), axis=0))

        restart_data_dict[i+1] = {
            'iter': restart_iter,
            'num_samples': num_samples,
            'random': random_data,
            'optim': optim_data,
            'best_val': best_val,
            'best_val_idx': best_val_idx,
        }

    restart_data_dict[0] = {
        'iter': 0,
        'num_samples': global_num_init,
        'random': (expr_vals[:global_num_init], expr_data[:global_num_init]),
        'optim': None,
        'best_val': np.min(expr_vals[:global_num_init]),
        'best_val_idx': np.argmin(expr_vals[:global_num_init]),
    }

    return expr_vals, expr_data, restart_data_dict


def extract_turbo_expr_info(benchmark_name, expr_name, idx=0):
    log_file = get_expr_file(benchmark_name, expr_name, idx, log=True)
    print(log_file)

    expr_vals, expr_data = read_data(benchmark_name, expr_name, idx)
    optim_config: dict = load_from_config(expr_name)

    # Split data by the restart iterations
    num_evals = len(expr_vals)
    num_init = optim_config['optimizer_params']['n_init']

    restart_iters = [0] + extract_split_idxes(benchmark_name, expr_name, idx, turbo=True)
    print(restart_iters)
    restart_data_dict = {}
    for i, restart_iter in enumerate(restart_iters):
        if restart_iter >= num_evals: break
        num_samples = restart_iters[i + 1] - restart_iter if i + 1 < len(restart_iters) else num_evals - restart_iter

        random_sample_data = expr_data[restart_iter: restart_iter + num_init]
        random_sample_vals = expr_vals[restart_iter: restart_iter + num_init]
        random_data = (random_sample_vals, random_sample_data)

        optim_sample_data = expr_data[restart_iter + num_init: restart_iter + num_samples]
        optim_sample_vals = expr_vals[restart_iter + num_init: restart_iter + num_samples]
        optim_data = (optim_sample_vals, optim_sample_data)

        num_samples = len(optim_sample_vals) + len(random_sample_vals)
        best_val = np.min(np.concatenate((optim_sample_vals, random_sample_vals), axis=0))
        best_val_idx = np.argmin(np.concatenate((optim_sample_vals, random_sample_vals), axis=0))

        restart_data_dict[i+1] = {
            'iter': restart_iter,
            'num_samples': num_samples,
            'random': random_data,
            'optim': optim_data,
            'best_val': best_val,
            'best_val_idx': best_val_idx,
        }
    return expr_vals, expr_data, restart_data_dict