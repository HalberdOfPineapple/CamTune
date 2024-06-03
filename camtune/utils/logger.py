import os
import time
import logging
from ConfigSpace import Configuration
from .paths import get_log_dir

LOG_IDX = 0
LOGGER: logging.Logger = None
EXPR_NAME: str = None

def get_log_filename(log_dir: str, expr_name: str, algo_optim: bool=False, seed: int=None):
    log_filename = os.path.join(log_dir, f'{expr_name}.log')
    if algo_optim:
        if seed is None:
            return 0, log_filename
        else:
            # extract the seed from the log file where the line is like: '2024-05-10 19:38:54,617 - INFO - 	seed: 11126'
            idx = 1
            while os.path.exists(log_filename):
                with open(log_filename, 'r') as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    if 'seed' in line:
                        seed_str = line.split(':')[-1].strip()
                        if seed_str == str(seed):
                            return idx-1, log_filename
                        break
                log_filename = os.path.join(log_dir, f'{expr_name}_{idx}.log')
                idx += 1
            return idx-1, log_filename
            

    idx = 1
    while os.path.exists(log_filename):
        with open(log_filename, 'r') as f:
            lines = '\n'.join(f.readlines())
            if not 'Elapsed time:' in lines:
                # The log file is not complete and we can directly overwrite it
                break
        # The log file exists and is complete. Create a new log file
        log_filename = os.path.join(log_dir, f'{expr_name}_{idx}.log')
        idx += 1

    print_log(f"[Logger] Saving logs to file: {log_filename}", print_msg=True)
    return idx-1, log_filename


def init_logger(expr_name: str, log_dir: str=None, algo_optim: bool=False, seed:int=None):
    # Setup logging
    print('=' * 80)
    print(f"[Logger] Initialize logger for experiment: {expr_name}")

    if log_dir is None: log_dir = get_log_dir()
    log_idx, log_filename = get_log_filename(log_dir, expr_name, algo_optim=algo_optim, seed=seed)

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, 'w') as f:
        f.write('')
    logging.basicConfig(filename=log_filename, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the logger
    global LOGGER, LOG_IDX
    LOGGER = logging.getLogger(expr_name)
    LOGGER.info('\n')
    LOG_IDX = log_idx

    return log_idx, log_filename

def get_log_idx():
    global LOG_IDX
    return LOG_IDX

def get_logger():
    global LOGGER
    return LOGGER

def print_log(msg: str, end:str='\n', level: str='info', print_msg: bool=False):
    global LOGGER
    if print_msg or LOGGER is None:
        print(msg, end=end)

    if LOGGER is not None:
        if level == 'debug':
            LOGGER.debug(msg)
        elif level == 'info':
            LOGGER.info(msg)
        elif level == 'warning':
            LOGGER.warning(msg)
        elif level == 'error':
            LOGGER.error(msg)
        elif level == 'critical':
            LOGGER.critical(msg)
        else:
            raise ValueError(f'Unknown log level: {level}')
    
def log_results_explain(
        best_config: Configuration,
        eval_func: callable,
        db_config: dict,
        tuning_time: float, 
):
    print_log("=" * 80, print_msg=True)
    print_log(f'[Tune] Final concrete execution with best configuration:', print_msg=True)

    start_time = time.perf_counter()
    # eval_result: dict = eval_func(best_config, exec_overwrite='pgbench')
    eval_result: dict = eval_func(best_config, exec_overwrite='raw')
    if eval_result['exec_success'] == False:
        print_log(f"[Tune] Final execution: Knob application failed.")
    conc_exec_time = time.perf_counter() - start_time
    elapsed_time = tuning_time + conc_exec_time

    conc_perf_name: str = db_config['conc_perf_name']
    conc_perf_unit: str = db_config['conc_perf_unit']
    perf = eval_result[conc_perf_name]

    print_log("=" * 80, print_msg=True)
    print_log(f"Best {conc_perf_name}: {perf} ({conc_perf_unit})", print_msg=True)
    print_log(f"Best config:", print_msg=True)
    for k, v in dict(best_config).items():
        print_log(f"\t{k}: {v}", print_msg=True)
    
    print_log(f"[Tune] Elapsed time in total: {elapsed_time:.2f} seconds", print_msg=True)

def log_results(
        best_config: Configuration, 
        best_Y: float, 
        tuning_time: float, 
        perf_name: str
):
    print_log("=" * 80, print_msg=True)
    print_log(f"Best {perf_name}: {best_Y}", print_msg=True)
    print_log(f"Best config:", print_msg=True)
    for k, v in dict(best_config).items():
        print_log(f"\t{k}: {v}", print_msg=True)
    
    print_log(f"[Tune] Elapsed time: {tuning_time:.2f} seconds", print_msg=True)