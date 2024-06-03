import os
import argparse
import torch
import yaml
from time import perf_counter
from typing import Callable

from tuner import Tuner
from camtune.optimizer.benchmarks import BaseBenchmark, BENCHMARK_MAP, EffectiveBenchmark
from camtune.utils import (init_logger, print_log, set_expr_paths,
                           OPTIM_CONFIG_DIR, get_log_dir, get_result_dir)

def main(config, expr_name, benchmark_name):
    set_expr_paths(expr_name, benchmark_name, algo_optim=True)
    log_idx, log_filename = init_logger(expr_name, algo_optim=True, seed=config['seed'])
    if benchmark_name not in BENCHMARK_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # --------------------------------------
    # Print config
    print_log("-" * 50, print_msg=True)
    print_log(f"[AlgoOptim] Configuration for {expr_name}:", print_msg=True)
    for k, v in config.items():
        if not isinstance(v, dict):
            print_log(f"\t{k}: {v}", print_msg=True)
        else:
            print_log(f"\t{k}:", print_msg=True)
            for k2, v2 in v.items():
                print_log(f"\t{k2}: {v2}", print_msg=True)
    print_log("-" * 50, print_msg=True)

    # --------------------------------------
    # Setup benchmark
    benchmark_cls, benchmark_params = BENCHMARK_MAP[benchmark_name]
    benchmark_params: dict
    if 'effective_dim' in benchmark_params:
        benchmark: BaseBenchmark = benchmark_cls(
            effective_dim=benchmark_params.pop('effective_dim'), 
            **benchmark_params
        )
    else:
        benchmark: BaseBenchmark = benchmark_cls(**benchmark_params)
    obj_func: Callable = benchmark.obj_func
    bounds: torch.Tensor = benchmark.bounds
    discrete_dims: list = benchmark.discrete_dims

    # --------------------------------------
    # Setup tuner
    config['use_default'] = False
    if 'mcts' in config['optimizer'].lower() and getattr(benchmark, 'mcts_params', None) is not None:
        for k, v in benchmark.mcts_params.items():
            config['optimizer_params'][k] = v
    tuner = Tuner(
        expr_name=expr_name,
        args=config,
        obj_func=obj_func,
        bounds=bounds,
        discrete_dims=discrete_dims,
    )

    #  --------------------------------------
    # Optimization
    start_time = perf_counter()

    with torch.no_grad():
        result_X, result_Y = tuner.tune()

    elapsed_time = perf_counter() - start_time

    # --------------------------------------
    if getattr(tuner.optimizer, 'get_original_data'):
        print_log("[AlgoOptim] Saving original data (within normalized space)...", print_msg=True)
        result_X, result_Y = tuner.optimizer.get_original_data()

    best_X = result_X[result_Y.argmax()].tolist()
    best_Y = result_Y.max().item()
    result_X, result_Y = result_X.detach().cpu().numpy(), result_Y.detach().cpu().numpy()
    
    if benchmark.negate:
        best_Y = -best_Y
        result_Y = -result_Y

    print_log("-" * 50, print_msg=True)
    print_log(f"Best X: {best_X}", print_msg=False)
    print_log(f"Best Y: {best_Y}", print_msg=True)
    print_log(f"Elapsed time: {elapsed_time:.2f} seconds", print_msg=True)

    data_file_name = f"{expr_name}{'' if log_idx == 0 else f'_{log_idx}'}_data.log"
    data_file_name = os.path.join(get_result_dir(), data_file_name)
    print_log(f"[AlgoOptim] Saving data to {data_file_name}...", print_msg=True)
    with open(data_file_name, 'w') as f:
        for x, fx in zip(result_X, result_Y):
            x = str(list(x))
            f.write(f"{fx}, {x}\n")
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--expr_name', '-e', default='random_1000')
    args.add_argument('--benchmark_name', '-b', default='ackley20d')
    args.add_argument('--num_evals', '-n', type=int, default=500)
    args.add_argument('--seed', '-s', type=int, default=None)
    args = args.parse_args()

    expr_name: str = args.expr_name
    benchmark_name = args.benchmark_name
    num_evals = args.num_evals
    seed: int = args.seed

    config_file_name = os.path.join(OPTIM_CONFIG_DIR, f'{expr_name}.yaml')
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)

    config['num_evals'] = num_evals if num_evals > 0 else 500
    config['seed'] = seed if seed is not None else config.get('seed', 0)
    if not expr_name.endswith(f'_{num_evals}'):
        expr_name = f"{expr_name}_{num_evals}"

    main(config, expr_name, benchmark_name)