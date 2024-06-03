import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple

from camtune.utils import print_log, QUERY_PATH_MAP, BENCHMARK_DIR, BENCHMARKS
from camtune.database.workloads import *

from .utils import (
    parse_pgbench_output, extract_total_cost, 
    run_command, run_sql_query, run_sql_from_file
)
from .connector import PostgresqlConnector
from .variables import *

DBCONFIG = get_db_config()


class PGBaseExecutor(ABC):
    @property
    def is_benchbase(self) -> bool:
        return self.benchmark.lower() in BENCHBASE_MODES

    @abstractmethod
    def run_benchmark(self):
        raise NotImplementedError("[PGBaseExecutor] run_benchmark is not implemented")
    
    @abstractmethod
    def do_vaccum_full(self) -> bool:
        raise NotImplementedError("[PGBaseExecutor] do_vaccum_full is not implemented")

    @abstractmethod
    def do_analyze(self) -> bool:
        raise NotImplementedError("[PGBaseExecutor] do_analyze is not implemented")
    
    @abstractmethod
    def clear_vm_cache(self) -> bool:
        raise NotImplementedError("[PGBaseExecutor] clear_vm_cache is not implemented")
    
    @abstractmethod
    def start_pg_default(self):
        raise NotImplementedError("[PGBaseExecutor] start_pg_default is not implemented")

    @abstractmethod
    def recover_default_config(self):
        raise NotImplementedError("[PGBaseExecutor] recover_default_config is not implemented")


class PostgreExecutor(PGBaseExecutor):
    def __init__(self, 
                 benchmark: str, 
                 exec_mode: str='raw', 
                 sysbench_mode: str=None,
                 benchmark_params: dict = None,
                 enable_timeout: bool=True,
    ):
        if benchmark.lower() not in BENCHMARKS:
            raise ValueError(f"[PGExecutor] Undefined Benchmark {benchmark}")

        self.exec_mode = exec_mode
        if self.exec_mode not in EXEC_MODES:
            raise ValueError(f"[PGExecutor] Unsupported execution mode {self.exec_mode}")

        self.benchmark = benchmark
        self.benchmark_params = benchmark_params

        self.sysbench_mode = sysbench_mode

        self.enable_timeout = enable_timeout

        self.db_conn = PostgresqlConnector()

        self.init_benchmark()
    
    def init_benchmark(self):
        if self.benchmark.upper() == 'SYSBENCH':
            sysbench_config: dict = SYSBENCH_WORKLOADS[self.sysbench_mode]
            if self.benchmark_params is not None:
                if 'exec_time' in self.benchmark_params:
                    sysbench_config['time'] = self.benchmark_params['exec_time']
            self.sysbench_config = sysbench_config

            script_path = os.path.join(SYSBENCH_SCRIPTS_DIR, sysbench_config['script'])
            self.sysbench_command = SYSBENCH_CMD_TMP.format(
                script=script_path, 
                db_name=DBCONFIG.db_name, db_user=DBCONFIG.db_user, db_pwd=DBCONFIG.db_pwd, db_host='localhost',
                tables=sysbench_config['tables'], table_size=sysbench_config['table_size'], time=sysbench_config['time'],
            )
        
    def clear_vm_cache(self):
        clear_cmd_1 = "sync"
        _, _, _, clear_cmd_1 = run_command(clear_cmd_1, sudo=True)
        print_log(f'[PGExecutor] Clearing VM cache with command = {clear_cmd_1}', print_msg=True)

        clear_cmd_2 = 'sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"'
        _, _, _, clear_cmd_2 = run_command(clear_cmd_2, sudo=True)
        print_log(f'[PGExecutor] Clearing VM cache with command = {clear_cmd_2}', print_msg=True)
    
    def do_vaccum_full(self):
        _, _, _, vacuum_cmd = run_sql_query(VACUUM_QUERY)
        print_log(f'[PGExecutor] Executing  with command = {vacuum_cmd}', print_msg=True)
    
    def do_analyze(self):
        _, _, _, analyze_cmd = run_sql_query(ANALYZE_QUERY)
        print_log(f'[PGExecutor] Executing  with command = {analyze_cmd}', print_msg=True)
    
    def get_timeout(self):
        if self.enable_timeout:
            return get_group_timeout(DBCONFIG.db_name), get_single_timeout(DBCONFIG.db_name)
        else:
            return float('inf'), 0
    
    # --------------------------------------------------------------------
    # PostgreSQL Server Management
    def start_pg_default(self):
        restart_cmd = RESTART_CMD_WSL if DBCONFIG.on_wsl else RESTART_CMD
        proc, stdout, stderr, cmd = run_command(restart_cmd, sudo=True)

        print_log(f"[PGExecutor] Restart PostgreSQL with default configuration (cmd: {cmd})")
        return proc.returncode, stdout, stderr, cmd

    def recover_default_config(self):
        cp_cmd = 'cp {} {}'.format(DBCONFIG.pg_default_conf, DBCONFIG.pg_conf)
        run_command(cp_cmd, as_postgres=True)

    # --------------------------------------------------------------------
    # Benchmark Execution
    def run_benchmark(self, exec_overwrite: str=None) -> Tuple[dict, list]:
        if self.benchmark.lower() == 'sysbench':
            return self.run_sysbench()
        elif self.benchmark.lower() == 'ycsba' or self.benchmark.lower() == 'ycsbb':
            return self.run_ycsb()

        res_dict = {'benchmark': self.benchmark}
        exec_mode = self.exec_mode if exec_overwrite is None else exec_overwrite
        query_file_names = self.load_query_paths(exec_mode)

        if exec_mode == 'pgbench':   
            total_exec_time = 0

            results = self.exec_queries_pgbench(query_file_names)
            for query_file_name in query_file_names:
                if query_file_name in results:
                    # latency measured in ms
                    avg_latency = float(results[query_file_name]['latency_average'])
                    total_exec_time += avg_latency

            res_dict['total_exec_time'] = total_exec_time
        elif exec_mode == 'raw':
            results = self.exec_queries(query_file_names)
            num_queries = len(query_file_names)
            if results is not None:
                total_exec_time = 0
                for i, (_, exec_res) in enumerate(results.items()):
                    if exec_res == SINGLE_QUERY_TIMEOUT: # if a single query times out, the total time is set to twice of the timeout
                        total_exec_time += get_single_timeout() * (num_queries - i)
                        break
                    elif exec_res == ACCUM_QUERY_TIMEOUT: # if the total time exceeds the timeout, the total time is set to the timeout
                        total_exec_time *= num_queries / (i + 1)
                        break
                    exec_time = float(exec_res[0]['execution_time'])
                    total_exec_time += exec_time
            else:
                raise ValueError(f"[PGExecutor] Connection failed when executing queries")

            res_dict['total_exec_time'] = total_exec_time
        elif exec_mode == 'explain':
            total_cost: float = 0

            results = self.exec_queries_explain(query_file_names)
            for i, (_, exec_res) in enumerate(results.items()):
                # print_log(f"[PGExecutor] Result of executing {query_file_names[i]}:\n{exec_res}")
                exec_plan: str = exec_res[0]['output']
                total_cost += extract_total_cost(exec_plan)

            res_dict['total_cost'] = total_cost
        else:
            raise ValueError(f"[PGExecutor] Unsupported execution mode {exec_mode}")

        return res_dict

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
        print_log(f"[PGExecutor] Exeucting queries listed in {query_list_file}")

        if benchmark == 'tpch' or benchmark == 'job':
            lines = open(os.path.join(BENCHMARK_DIR, query_list_file), 'r').readlines()
            for line in lines:
                query_file_names.append(line.rstrip())
            query_file_names = [os.path.join(self.query_dir, query_file_name) for query_file_name in query_file_names]
        else:
            raise ValueError(f"[PGExecutor] Unsupported benchmark {self.benchmark} when loading query paths")
    
        return query_file_names

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

        results = {}
        # Executing queries and fetch execution results
        group_timeout, single_timeout = self.get_timeout()
        total_exec_time = 0
        for query_file, query in queries:
            try:
                start_time = time.perf_counter()
                out, err = run_sql_from_file(query_file, timeout=single_timeout)
                exec_time = time.perf_counter() - start_time

                total_exec_time += exec_time
                print_log(f'[PGExecutor] Executing {query_file} takes {exec_time:.3f}s.')
                if total_exec_time > group_timeout:
                    results[query_file] = ACCUM_QUERY_TIMEOUT
                    break

                results[query_file] = [{
                    'execution_time': exec_time,
                    'output': out,
                }]
            except KeyboardInterrupt:
                print_log(f'[PGExecutor] Query execution interrupted from active CTRL-C when executing {query_file}')
                exit(1)
            except subprocess.TimeoutExpired as e:
                print_log(f"[PGConnector] Executing {query_file} exceeded the time limit {single_timeout}s.", print_msg=True)
                results[query_file] = SINGLE_QUERY_TIMEOUT
                break
            except Exception as e:
                print_log(f'[PGExecutor] Query execution failed when executing {query_file}')
                print_log(f"[PGExecutor] Error information: {e}\t{traceback.format_exc()}")      

        return results
    

    # Assume the query files are stored locally (to be executed remotely)
    def exec_queries_explain(self, query_file_names, json: bool=True):
        queries = []
        for query_file in query_file_names:
            with open(query_file, 'r') as f:
                query_lines = f.readlines()
            query = ' '.join(query_lines)
            queries.append((query_file, query))

        results = {}
        for query_file, query in queries:
            print_log(f'[PGExecutor] Executing {query_file}')
            try:
                result = self.db_conn.fetch_results(query, json=json)
                results[query_file] = result
            except Exception as e:
                print_log((
                    f'[PGExecutor] Query execution failed when executing {query_file},'
                    f' with error information: {e}'
                ))

        self.db_conn.close_db()
        return results

    # --------------------------------------------------------
    # Sysbench
    # --------------------------------------------------------
    def run_sysbench(self):
        print_log(f"[PGExecutor] Running sysbench with command: {self.sysbench_command}", print_msg=True)
        proc, stdout, stderr, cmd = run_command(self.sysbench_command, as_postgres=True)
        result_dict = parse_sysbench_output(stdout)
        return result_dict

    # --------------------------------------------------------
    # YCSB
    # --------------------------------------------------------
    def run_ycsb(self):
        ycsb_config: dict = YCSB_WORKLOADS[self.ycsb_mode]
        ycsb_workload = os.path.join(YCSB_WORKLOAD_DIR, ycsb_config['workload'])
        ycsb_prop = ycsb_config['prop_path']
        ycsb_cmd = (f"{YCSB_BIN} run postgrenosql -p operationcount=100000 -P {ycsb_workload}"
                    f" -P {ycsb_prop} -p maxexecutiontime={ycsb_config['time']}")
        print_log(f"[PGExecutor] Running YCSB with command: {ycsb_cmd}")

        proc, stdout, stderr, cmd = run_command(ycsb_cmd)
        result_dict: dict = parse_ycsb_output(stdout)

        return result_dict

    # --------------------------------------------------------
    # PgBench
    # --------------------------------------------------------
    def exec_queries_pgbench(self, query_file_names: List[str]):
        res_dict: dict = {}
        for query_file_name in query_file_names:
            print_log(f"[Executor] Executing {query_file_name} using pgbench")
            command = f"pgbench -f {query_file_name} {DBCONFIG.db_name} -n"
            proc, stdout, stderr, cmd = run_command(command)
            if proc.returncode != 0:
                print_log(f'[PGExecutor] Local execution of SQL query {query_file_name} using pgbench failed')
            res_dict[query_file_name] = parse_pgbench_output(stdout)
        return res_dict