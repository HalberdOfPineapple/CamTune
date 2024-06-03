import os

EXPR_NAME = None
BENCHMARK_NAME = None
ALGO_OPTIM = False
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, 'figs')
TEMP_DIR = os.path.join(BASE_DIR, 'tmp')
TR_KNOB_DIR = os.path.join(BASE_DIR, 'tr_knobs')

CONFIG_DIR = os.path.join(BASE_DIR, 'config')
KNOB_DIR = os.path.join(BASE_DIR, 'knobs')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
TREE_DIR = os.path.join(BASE_DIR, 'trees')
LLAMA_LOG_DIR = os.path.join(BASE_DIR, 'logs_llama')

OPTIM_CONFIG_DIR = os.path.join(BASE_DIR, 'optimizer', 'optim_config')
OPTIM_LOG_DIR = os.path.join(BASE_DIR, 'optimizer', 'optim_logs')
OPTIM_RES_DIR = os.path.join(BASE_DIR, 'optimizer', 'optim_results')
OPTIM_TREE_DIR = os.path.join(BASE_DIR, 'optimizer', 'optim_trees')

BENCHMARK_DIR = os.path.join(BASE_DIR, 'benchmarks')
QUERY_PATH_MAP = {
    'TPCH': os.path.join(BENCHMARK_DIR, 'tpch'),
    'JOB': os.path.join(BENCHMARK_DIR, 'job'),
}

paths = [CONFIG_DIR, LOG_DIR, OPTIM_CONFIG_DIR, OPTIM_LOG_DIR, BENCHMARK_DIR, TEMP_DIR]
for path in paths:
    if not os.path.exists(path):
        # recursively create the path
        os.makedirs(path)

def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_log_dir(log_dir):
    global LOG_DIR
    LOG_DIR = log_dir
    check_and_create_dir(LOG_DIR)
    
def set_result_dir(result_dir):
    global RESULT_DIR
    RESULT_DIR = result_dir
    check_and_create_dir(RESULT_DIR)

def set_tree_dir(tree_dir):
    global TREE_DIR
    TREE_DIR = tree_dir
    check_and_create_dir(TREE_DIR)

def set_expr_paths(expr_name, benchmark_name, dummy_exec: bool =False, algo_optim: bool=False):
    global EXPR_NAME, BENCHMARK_NAME, ALGO_OPTIM
    EXPR_NAME = expr_name
    BENCHMARK_NAME = benchmark_name
    ALGO_OPTIM = algo_optim

    if algo_optim:
        global LOG_DIR, RESULT_DIR, TREE_DIR
        LOG_DIR = OPTIM_LOG_DIR
        RESULT_DIR = OPTIM_RES_DIR
        TREE_DIR = OPTIM_TREE_DIR
        
    if dummy_exec:
        set_log_dir(os.path.join(LOG_DIR, 'dummy', benchmark_name))
        set_result_dir(os.path.join(RESULT_DIR, 'dummy', benchmark_name))
        set_tree_dir(os.path.join(TREE_DIR, 'dummy', benchmark_name))
    else:
        set_log_dir(os.path.join(LOG_DIR, benchmark_name))
        set_result_dir(os.path.join(RESULT_DIR, benchmark_name))
        set_tree_dir(os.path.join(TREE_DIR, benchmark_name))

def get_expr_name():
    return EXPR_NAME

def get_benchmark_name() -> str:
    return BENCHMARK_NAME

def get_log_dir():
    return LOG_DIR

def get_result_dir():
    return RESULT_DIR

def get_tree_dir():
    return TREE_DIR