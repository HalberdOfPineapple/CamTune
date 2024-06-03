import os
import argparse
from camtune.utils import (
    set_expr_paths, get_log_dir, get_result_dir, get_tree_dir,
)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--expr_name', '-e', default='random_1000')
    args.add_argument('--benchmark_name', '-b', default='ackley20d')
    args.add_argument('--idx', '-i', default=0, type=int)
    args.add_argument('--algo_optim', '-a', action='store_true')
    args = args.parse_args()

    expr_name = args.expr_name
    benchmark_name = args.benchmark_name
    algo_optim = args.algo_optim
    idx = args.idx

    set_expr_paths(expr_name, benchmark_name, algo_optim=algo_optim)
    
    log_file_name = os.path.join(get_log_dir(), f'{expr_name}_{idx}.log') if idx > 0 else os.path.join(get_log_dir(), f'{expr_name}.log')
    res_file_name = os.path.join(get_result_dir(), f'{expr_name}_{idx}_data.json') if idx > 0 else os.path.join(get_result_dir(), f'{expr_name}_data.json')
    data_file_name = os.path.join(get_result_dir(), f'{expr_name}_{idx}_data.log') if idx > 0 else os.path.join(get_result_dir(), f'{expr_name}_data.log')
    tree_dir = os.path.join(get_tree_dir(), expr_name, f'{idx}')

    for file_name in [log_file_name, res_file_name, data_file_name]:
        # Delete the file/directories above
        if os.path.exists(file_name):
            print(f"Deleting file: {file_name}")
            os.remove(file_name)
        else:
            print(f"file does not exist: {file_name}")

    if 'winter' in expr_name or 'mcts' in expr_name:
        if os.path.exists(tree_dir):
            print(f"Deleting tree directory: {tree_dir}")
            os.rmdir(tree_dir)
        else:
            print(f"tree directory does not exist: {tree_dir}")


    

