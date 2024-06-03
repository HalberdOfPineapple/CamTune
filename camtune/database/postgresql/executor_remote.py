import os
import time
import traceback
import numpy as np
from typing import List, Tuple

from camtune.utils import (
    QUERY_PATH_MAP, BENCHMARK_DIR, BENCHMARKS, 
    print_log, get_result_dir, get_expr_name, get_log_idx)
from camtune.database.workloads import *

from .variables import *
from .utils import parse_pgbench_output, extract_total_cost, run_command
from .executor import PGBaseExecutor
from .ssh_client import PGSSHClient, parse_timing_result

DBCONFIG = get_db_config()

class PostgreRemoteExecutor(PGBaseExecutor):
    def __init__(
        self, 
        benchmark: str, 
        exec_mode: str='raw', 
        sysbench_mode: str=None,
        benchmark_params: dict = None,
        enable_timeout: bool=True,
    ):
        self.ssh_cli = PGSSHClient()

        if benchmark.lower() not in BENCHMARKS:
            raise ValueError(f"[PGRemoteExecutor] Undefined Benchmark {benchmark}")

        self.exec_mode = exec_mode
        if self.exec_mode not in EXEC_MODES:
            raise ValueError(f"[PGRemoteExecutor] Unsupported execution mode {self.exec_mode}")

        self.benchmark = benchmark
        self.benchmark_params = benchmark_params

        self.sysbench_mode = sysbench_mode

        self.timeout = get_group_timeout(DBCONFIG.db_name)
        self.enable_timeout = enable_timeout

        self.init_benchmark()
    
    def init_benchmark(self):
        if self.benchmark.upper() == 'SYSBENCH':
            sysbench_config: dict = SYSBENCH_WORKLOADS[self.sysbench_mode]
            if self.benchmark_params is not None:
                if 'exec_time' in self.benchmark_params:
                    sysbench_config['time'] = self.benchmark_params['exec_time']
            self.sysbench_config = sysbench_config

            self.sysbench_timeout = sysbench_config['time'] + 100
            script_path = os.path.join(SYSBENCH_SCRIPTS_DIR, sysbench_config['script'])
            self.sysbench_command = SYSBENCH_CMD_TMP.format(
                script=script_path, 
                db_name=DBCONFIG.db_name, db_user=DBCONFIG.db_user, db_pwd=DBCONFIG.db_pwd, db_host='localhost',
                tables=sysbench_config['tables'], table_size=sysbench_config['table_size'], time=sysbench_config['time'],
            )

            if 'postfix' in sysbench_config:
                self.sysbench_command += f" {sysbench_config['postfix']}"

        elif self.benchmark.lower() in BENCHBASE_MODES:
            # Init res idx
            self.res_dir_idx = 0
            self.benchmark_name: str = get_benchmark_name(self.benchmark)

            # Try to drop the database beforehand (regardless whether it exists or not)
            drop_db_cmd = f"dropdb {DBCONFIG.db_name}"
            print_log(f"[PGRemoteExecutor] Drop database for BenchBase with command: {drop_db_cmd}")
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(drop_db_cmd)

            # Create database for BenchBase
            create_db_cmd = f"createdb {DBCONFIG.db_name}"
            print_log(f"[PGRemoteExecutor] Create database for BenchBase with command: {create_db_cmd}")
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(create_db_cmd)
            if ret_code != 0:
                raise RuntimeError(f"[PGRemoteExecutor] Failed to create database for BenchBase: {stderr}")

            # Synchronize BenchBase configuration file
            self.benchbase_config_path = os.path.join(BENCHBASE_CONFIG_DIR, f'{self.benchmark.lower()}.xml')
            print_log(f"[PGRemoteExecutor] Synchronize BenchBase configuration path: {self.benchbase_config_path} with remote host")
            self.ssh_cli.sync_file(self.benchbase_config_path, self.benchbase_config_path)

            # Synchronize BenchBase shell script
            print_log(f"[PGRemoteExecutor] Synchronize BenchBase script {BENCHBASE_SCRIPT} with remote host")
            self.ssh_cli.sync_file(BENCHBASE_SCRIPT, BENCHBASE_SCRIPT)

            # Synchronize BenchBase load script
            print_log(f"[PGRemoteExecutor] Synchronize BenchBase load script {BENCHBASE_LOAD_SCRIPT} with remote host")
            self.ssh_cli.sync_file(BENCHBASE_LOAD_SCRIPT, BENCHBASE_LOAD_SCRIPT)
            
            # Create directory for BenchBase result remotely
            benchbase_res_dir = os.path.join(get_result_dir(), f'{get_expr_name()}_{get_log_idx()}')
            dir_delete_cmd = f"rm -rf {benchbase_res_dir}"
            run_command(dir_delete_cmd)
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(dir_delete_cmd)
            if ret_code != 0:
                raise RuntimeError(f"[PGRemoteExecutor] Failed to delete directory for BenchBase result: {stderr}")

            # Create directory for BenchBase result remotely
            dir_create_cmd = f"mkdir -p {benchbase_res_dir}"
            run_command(dir_create_cmd)
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(dir_create_cmd)
            if ret_code != 0:
                raise RuntimeError(f"[PGRemoteExecutor] Failed to create directory for BenchBase result: {stderr}")
            self.benchbase_res_dir = benchbase_res_dir

            # Load BenchBase data
            load_res_dir = os.path.join(benchbase_res_dir, 'load')
            print_log(f"[PGRemoteExecutor] Loading Benchbase data for {self.benchmark_name}...", print_msg=True)
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(f"mkdir -p {load_res_dir}")
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(
                f"bash {BENCHBASE_LOAD_SCRIPT} {self.benchmark_name} {self.benchbase_config_path} {load_res_dir}")
            print_log(f"[PGRemoteExecutor] BenchBase data loaded with command: {cmd}", print_msg=True)
            if ret_code != 0:
                raise RuntimeError(f"[PGRemoteExecutor] Failed to load BenchBase data: stdout: \n{stdout} \nstrerr: \n{stderr}")

    
    
    # ------------------------------------------------------------------
    # PostgreSQL Operations
    def start_pg_default(self):
        start_command = "systemctl restart postgresql"
        print_log(f"[PGRemoteExecutor] Restart PostgreSQL with default configuration (cmd: {start_command})")
        return self.ssh_cli.remote_exec_command(start_command, sudo=True, timeout=120)

    def recover_default_config(self):
        cp_cmd = 'cp {} {}'.format(DBCONFIG.pg_default_conf, DBCONFIG.pg_conf)
        self.ssh_cli.remote_exec_command(cp_cmd, as_postgre=True)

    def clear_vm_cache(self):
        clear_cmd_1 = "sync"
        ret_code, std_out, std_err, clear_cmd_1 = self.ssh_cli.remote_exec_command(clear_cmd_1, sudo=True)
        print_log(f'[PGRemoteExecutor] Clearing VM cache with command = {clear_cmd_1}', print_msg=True)
        if ret_code != 0:
            print_log(f'[PGRemoteExecutor] Failed to execute {clear_cmd_1} with output: {std_out} and with error: {std_err}', print_msg=True)
            return False

        clear_cmd_2 = 'sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"'
        ret_code, std_out, std_err, clear_cmd_2 = self.ssh_cli.remote_exec_command(clear_cmd_2, sudo=True)
        print_log(f'[PGRemoteExecutor] Clearing VM cache with command = {clear_cmd_2}', print_msg=True)
        if ret_code != 0:
            print_log(f'[PGRemoteExecutor] Failed to execute {clear_cmd_2} with output: {std_out} and with error: {std_err}', print_msg=True)
            return False
        return True
    
    def do_vaccum_full(self):
        print_log(f'[PGRemoteExecutor] Executing vaccumming with SQL = {VACUUM_QUERY}', print_msg=True)
        ret_code, std_out, std_err, vacuum_cmd = self.ssh_cli.remote_exec_sql(VACUUM_QUERY)
        if ret_code != 0:
            print_log(f'[PGRemoteExecutor] Failed to execute {vacuum_cmd} with output: {std_out} and error: {std_err}', print_msg=True)
            return False
        return True

    def do_analyze(self):
        print_log(f'[PGRemoteExecutor] Executing vaccumming with SQL = {ANALYZE_QUERY}', print_msg=True)
        ret_code, std_out, std_err, analyze_cmd = self.ssh_cli.remote_exec_sql(ANALYZE_QUERY)
        if ret_code != 0:
            print_log(f'[PGRemoteExecutor] Failed to execute {analyze_cmd} with output: {std_out} and error: {std_err}', print_msg=True)
            return False
        return True

    def reboot(self):
        self.ssh_cli.reboot()


    # ---------------------------------------------------------------
    # Benchmark Execution Related
    def load_query_paths(self, exec_mode: str):
        # Get the directory to the query files
        self.query_dir = QUERY_PATH_MAP[self.benchmark.upper()]
        if exec_mode == 'pgbench':
            self.query_dir = self.query_dir + '_pgbench'
        elif exec_mode == 'explain':
            self.query_dir = self.query_dir + '_explain'
        else:
            self.query_dir = self.query_dir + '_raw'
        
        # Get the list of exact query file names
        query_file_names = []
        benchmark = self.benchmark.lower()
        query_list_file = f'{benchmark}_query_list.txt'
        print_log(f"[PGRemoteExecutor] Exeucting queries listed in {query_list_file}")

        if benchmark == 'tpch' or benchmark == 'job':
            lines = open(os.path.join(BENCHMARK_DIR, query_list_file), 'r').readlines()
            for line in lines:
                query_file_names.append(line.rstrip())
            query_file_names = [os.path.join(self.query_dir, query_file_name) for query_file_name in query_file_names]
        else:
            raise ValueError(f"[PGRemoteExecutor] Unsupported benchmark {self.benchmark} when loading query paths")
    
        return query_file_names
    
    def run_benchmark(self, exec_overwrite: str=None) -> Tuple[dict, list]:
        if self.benchmark.lower() == 'sysbench':
            return self.run_sysbench()
        elif self.benchmark.lower() in BENCHBASE_MODES:
            return self.run_benchbase()

        res_dict = {}
        exec_mode = self.exec_mode if exec_overwrite is None else exec_overwrite
        query_file_names = self.load_query_paths(exec_mode)

        if exec_mode == 'pgbench':   
            return self.exec_queries_pgbench(query_file_names)
        elif exec_mode == 'raw':
            return self.exec_queries(query_file_names)
        elif exec_mode == 'explain':
            return self.exec_queries_explain(query_file_names)
        else:
            raise ValueError(f"[PGRemoteExecutor] Unsupported execution mode {exec_mode}")

    # --------------------------------------------------------
    # Raw Queries
    # --------------------------------------------------------
    # Assume the query files are stored locally (to be executed remotely)
    def exec_queries(self, query_file_names: List[str]):
        queries = []
        for query_file in query_file_names:
            with open(query_file, 'r') as f:
                query_lines = f.readlines()
            query = ' '.join(query_lines)
            queries.append((query_file, query))

        res_dict, exec_time_map = {}, {}
        total_exec_time = 0
        final_idx = len(query_file_names)
        exec_err = False
        for i, (query_file, query) in enumerate(queries):
            try:
                left_time = self.timeout - total_exec_time if self.enable_timeout else None

                
                ret_code, out, err, _ = self.ssh_cli.remote_exec_sql_from_local_file(
                                        query_file, timing=True, timeout=left_time)
                if ret_code == 124: 
                    print_log(f"[PGRemoteExecutor] Query execution timeout for {query_file}")
                    exec_time_map[query_file] = self.timeout
                    final_idx = i
                    break
                elif ret_code != 0 or out is None or len(out) == 0:
                    print_log(f"[PGRemoteExecutor] Failed to execute query {query_file}: {err[:5000]}")
                    exec_time_map[query_file] = self.timeout * 2
                    exec_err = True
                    break
                else:
                    exec_time = parse_timing_result(err)
                    if exec_time < 0:
                        print_log(f"[PGRemoteExecutor] Failed to get execution time from the timing result for query {query_file}: {err[:5000]}")
                        exec_time = self.timeout * 2
                    print_log(f'[PGRemoteExecutor] Executing {query_file} takes {exec_time:.2f}s')

                    exec_time_map[query_file] = exec_time
                    total_exec_time += exec_time
                    if total_exec_time > self.timeout:
                        final_idx = i
                        break
            except KeyboardInterrupt:
                print_log(f'[PGRemoteExecutor] Query execution interrupted from active CTRL-C when executing {query_file}')
                exit(1)
    
        if not exec_err:
            if final_idx < len(query_file_names):
                for i in range(final_idx, len(query_file_names)):
                    exec_time_map[query_file_names[i]] = self.timeout
                    total_exec_time += self.timeout
            exec_time_list = list(exec_time_map.values())

            res_dict['latency'] = np.percentile(exec_time_list, 95)
            res_dict['execution_time'] = exec_time_map
            res_dict['total_exec_time'] = total_exec_time
        else:
            res_dict['latency'] = self.timeout * 2
            res_dict['execution_time'] = {query_file: self.timeout * 2 for query_file in query_file_names}
            res_dict['total_exec_time'] = self.timeout * 2 * len(query_file_names)

        return res_dict
    
    # --------------------------------------------------------
    # Explain
    # --------------------------------------------------------
    # Assume the query files are stored locally (to be executed remotely)
    def exec_queries_explain(self, query_file_names: List[str], json: bool=True):
        queries = []
        for query_file in query_file_names:
            with open(query_file, 'r') as f:
                query_lines = f.readlines()
            query = ' '.join(query_lines)
            queries.append((query_file, query))

        res_dict, results = {}, {}
        for query_file, query in queries:
            print_log(f'[PGRemoteExecutor] Executing {query_file}')
            try:
                _, std_out, _, _ = self.ssh_cli.remote_exec_sql(query)
                results[query_file] = std_out
            except Exception as e:
                print_log((
                    f'[PGRemoteExecutor] Query (Explain) execution failed when executing {query_file}, '
                    f'with error information: {e}\t{traceback.format_exc()}'
                ))  
        total_cost: float = 0
        for i, (_, exec_res) in enumerate(results.items()):
            # print_log(f"[PGExecutor] Result of executing {query_file_names[i]}:\n{exec_res}")
            exec_plan: str = exec_res[0]['output']
            total_cost += extract_total_cost(exec_plan)

        res_dict['total_cost'] = total_cost
        return results

    # --------------------------------------------------------
    # BenchBase
    # --------------------------------------------------------
    def run_benchbase(self):
        benchbase_res_dir = os.path.join(self.benchbase_res_dir, str(self.res_dir_idx))
        self.res_dir_idx += 1

        ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(f"mkdir -p {benchbase_res_dir}")
        if ret_code != 0:
            raise RuntimeError(f"[PGRemoteExecutor] Failed to create directory for BenchBase result: {stderr}")

        
        benchbase_cmd = \
            f"bash {BENCHBASE_SCRIPT} {self.benchmark_name} {self.benchbase_config_path} {benchbase_res_dir}"

        print_log(f"[PGRemoteExecutor] BenchBase execution with command: {benchbase_cmd}")
        ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(benchbase_cmd, timeout=BENCHBASE_TIMEOUT)
        if ret_code != 0:
            raise RuntimeError(f"[PGRemoteExecutor] BenchBase execution failed with output: {stdout[:5000]} and error: {stderr}")
    
        self.ssh_cli.sync_from_remote_dir(benchbase_res_dir, benchbase_res_dir)
        return parse_benchbase(benchbase_res_dir)

    def drop_benchbase_data(self):
        drop_cmd = f"dropdb {DBCONFIG.db_name}"
        print_log(f"[PGRemoteExecutor] Dropping BenchBase data with SQL: {drop_cmd}", print_msg=True)
        ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(drop_cmd)
        if ret_code != 0:
            raise RuntimeError(f"[PGRemoteExecutor] Failed to drop BenchBase data: {stderr}")

    # --------------------------------------------------------
    # Sysbench
    # --------------------------------------------------------
    def run_sysbench(self):
        ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(self.sysbench_command, as_postgre=True, timeout=self.sysbench_timeout)
        print_log(f"[PGRemoteExecutor] Running sysbench with command: {cmd}")
        if ret_code != 0:
            raise RuntimeError(f"[PGRemoteExecutor] SysBench execution failed with output: {stdout[:5000]} and error: {stderr}")

        return parse_sysbench_output(stdout)

    # --------------------------------------------------------
    # PgBench TODO
    # --------------------------------------------------------
    def exec_queries_pgbench(self, query_file_names: List[str]):
        results: dict = {}
        for query_file_name in query_file_names:
            print_log(f"[Executor] Executing {query_file_name} using pgbench")
            command = f"pgbench -f {query_file_name} {DBCONFIG.db_name} -n"
            ret_code, stdout, stderr, cmd = self.ssh_cli.remote_exec_command(command, as_postgre=True)
            if ret_code != 0:
                print_log(f'[PGRemoteExecutor] Local execution of SQL query {query_file_name} using pgbench failed')
            results[query_file_name] = parse_pgbench_output(stdout)
        
        total_exec_time = 0
        for query_file_name in query_file_names:
            if query_file_name in results:
                # latency measured in ms
                avg_latency = float(results[query_file_name]['latency_average'])
                total_exec_time += avg_latency

        res_dict = {'total_exec_time': total_exec_time}
        return res_dict
    