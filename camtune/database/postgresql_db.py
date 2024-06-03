import os
import random
import traceback
from time import perf_counter
from ConfigSpace.configuration_space import Configuration 

from camtune.database.utils import initialize_knobs
from camtune.database.postgresql import (
    init_global_vars,
    PGBaseExecutor, PostgreExecutor, PostgreRemoteExecutor, PostgreKnobApplier
)
from camtune.database.postgresql.variables import *
from camtune.utils import init_logger, print_log, KNOB_DIR, LLAMA_LOG_DIR
from camtune.database.workloads import BENCHBASE_MODES

from .base_database import BaseDatabase
DBCONFIG = get_db_config()


class PostgresqlDB(BaseDatabase):
    def __init__(self, args_db: dict, llama_params: dict = None, on_wsl: bool = False):
        self.args_db = args_db

        if llama_params:
            expr_name = llama_params['expr_name']
            init_logger(expr_name, args_db['benchmark'] ,log_dir=LLAMA_LOG_DIR)

        # ---------------- Connection & Server Settings --------------
        init_global_vars(args_db, on_wsl=on_wsl)
        print_log('=' * 50, print_msg=True)
        print_log(f"[PostgresqlDB] Initialized with: ", print_msg=True)
        print_log(f"{DBCONFIG}", print_msg=True)
        print_log('=' * 50, print_msg=True)

        # ------------------ Mode Settings -----------------------
        self.remote_mode: bool = args_db.get('remote_mode', False) if DBCONFIG.db_host == 'localhost' else True
        self.online_mode: bool = args_db.get('online_mode', False)

        # ------------------ Workload Settings -----------------------
        # Note that query can be saved locally
        self.executor: PGBaseExecutor
        if self.remote_mode:
            self.executor = PostgreRemoteExecutor(
                benchmark=args_db['benchmark'],
                exec_mode=args_db.get('exec_mode', 'raw'),
                sysbench_mode=args_db.get('sysbench_mode', None),
                benchmark_params=args_db.get('benchmark_params', None),
                enable_timeout=args_db.get('enable_timeout', True),
            )
        else:
            self.executor = PostgreExecutor(
                benchmark=args_db['benchmark'],
                exec_mode=args_db.get('exec_mode', 'raw'),
                sysbench_mode=args_db.get('sysbench_mode', None),
                benchmark_params=args_db.get('benchmark_params', None),
                enable_timeout=args_db.get('enable_timeout', True),
            )
        
        # ------------------ Knob Settings -----------------------
        knob_definition_path = os.path.join(KNOB_DIR, args_db['knob_definitions'])
        self.knob_details = initialize_knobs(knob_definition_path, args_db['knob_num'])
        self.knob_applier = PostgreKnobApplier(
            remote_mode=self.remote_mode,
            knob_details=self.knob_details,
            ssh_cli=self.executor.ssh_cli if self.remote_mode else None,
        )
        self.last_call_time = perf_counter()
    
    # ------------------------------------------------------------------
    def reboot(self):
        if self.remote_mode:
            self.executor: PostgreRemoteExecutor = self.executor
            self.executor.reboot()

    # ------------------------------------------------------------------
    def clear_sys_states(self):
        success: bool = self.executor.clear_vm_cache()
        if success:
            print_log(f"[PostgresqlDB] VM cache cleared.", print_msg=True)

    # ------------------------------------------------------------------
    def clear_db_states(self, clear_db_data: bool = False):
        if self.executor.is_benchbase:
            self.executor.drop_benchbase_data()
        elif clear_db_data:
            self.do_vaccum_full()

    def do_vaccum_full(self):
        if self.executor.do_vaccum_full():
            print_log(f"[PostgresqlDB] VACUUM Full completed.", print_msg=True)
        
        if self.executor.do_analyze():
            print_log(f"[PostgresqlDB] ANALYZE completed.", print_msg=True)
        
    # ------------------------------------------------------------------
    def default_restart(self, exec: bool =False, dummy: bool = False):
        self.executor.recover_default_config()
        
        success, err_msg = self.knob_applier.kill_postgres()
        if not success:
            raise RuntimeError(f"[PostgresqlDB] Error in default restart (when terminating PG server) with error: {err_msg}")
        success, err_msg = self.knob_applier.start_postgres()
        if not success:
            raise RuntimeError(f"[PostgresqlDB] Error in default restart (when launching PG server) with error: {err_msg}")

        if exec:
            self.step(config=None, dummy=dummy)

    # ------------------------------------------------------------------
    def apply_knob(self, config: Configuration, dummy: bool = False):
        knobs = dict(config).copy()
        knob_applied, err_msg = self.knob_applier.apply_knobs(knobs, self.online_mode)
        return knob_applied, err_msg

    # ------------------------------------------------------------------
    def step(self, config: Configuration, exec_overwrite: str = None, dummy: bool = False) -> dict:
        optim_exec_time = perf_counter() - self.last_call_time
        prep_time, bench_exec_time = 0, 0

        prep_start_time = perf_counter()
        self.executor.recover_default_config()
        try:
            if dummy:
                res = {self.args_db['perf_name']: random.random()}, []
                res['exec_success'] = True
                return res

            
            if config:
                print_log("-" * 80)
                knob_applied, err_msg = self.apply_knob(config)
                print_log('-' * 80)
            else:
                knob_applied, err_msg = True, None
                print_log(f'[PostgresqlDB] Default configuration applied.')
            prep_time = perf_counter() - prep_start_time
            
            bench_start_time = perf_counter()
            if knob_applied:
                res = self.executor.run_benchmark(exec_overwrite=exec_overwrite)
                res['exec_success'] = True
            else:
                res = {'exec_success': False, 'exec_error': err_msg}
            bench_exec_time = perf_counter() - bench_start_time

        except Exception as e:
            print_log(f"[PostgresqlDB] Error in step: {e}\t{traceback.format_exc()}")
            res = {'exec_success': False, 'exec_error': str(e)}
        
        res['optim_exec_time'] = optim_exec_time
        res['prep_time'] = prep_time
        res['bench_exec_time'] = bench_exec_time

        self.last_call_time = perf_counter()
        return res



# class PostgresqlDB:
#     def __init__(self, args_db: dict, llama_params: dict = None, on_wsl: bool = False):
#         self.args_db = args_db

#         if llama_params:
#             expr_name = llama_params['expr_name']
#             init_logger(expr_name, args_db['benchmark'] ,log_dir=LLAMA_LOG_DIR)

#         # ---------------- Connection & Server Settings --------------
#         init_global_vars(args_db, on_wsl=on_wsl)
#         print_log('=' * 50, print_msg=True)
#         print_log(f"[PostgresqlDB] Initialized with: ", print_msg=True)
#         print_log(f"{DBCONFIG}", print_msg=True)
#         print_log('=' * 50, print_msg=True)

#         # ------------------ Mode Settings -----------------------
#         self.remote_mode: bool = args_db.get('remote_mode', False) if DBCONFIG.db_host == 'localhost' else True
#         self.online_mode: bool = args_db.get('online_mode', False)

#         # ------------------ Workload Settings -----------------------
#         # Note that query can be saved locally
#         self.executor: PGBaseExecutor
#         if self.remote_mode:
#             self.executor = PostgreRemoteExecutor(
#                 benchmark=args_db['benchmark'],
#                 exec_mode=args_db.get('exec_mode', 'raw'),
#                 sysbench_mode=args_db.get('sysbench_mode', None),
#                 benchmark_params=args_db.get('benchmark_params', None),
#                 enable_timeout=args_db.get('enable_timeout', True),
#             )
#         else:
#             self.executor = PostgreExecutor(
#                 benchmark=args_db['benchmark'],
#                 exec_mode=args_db.get('exec_mode', 'raw'),
#                 sysbench_mode=args_db.get('sysbench_mode', None),
#                 benchmark_params=args_db.get('benchmark_params', None),
#                 enable_timeout=args_db.get('enable_timeout', True),
#             )
        
#         # ------------------ Knob Settings -----------------------
#         knob_definition_path = os.path.join(KNOB_DIR, args_db['knob_definitions'])
#         self.knob_details = initialize_knobs(knob_definition_path, args_db['knob_num'])
#         self.knob_applier = PostgreKnobApplier(
#             remote_mode=self.remote_mode,
#             knob_details=self.knob_details,
#             ssh_cli=self.executor.ssh_cli if self.remote_mode else None,
#         )
#         self.last_call_time = perf_counter()


#     def do_vaccum_full(self):
#         if self.executor.do_vaccum_full():
#             print_log(f"[PostgresqlDB] VACUUM Full completed.", print_msg=True)
        
#         if self.executor.do_analyze():
#             print_log(f"[PostgresqlDB] ANALYZE completed.", print_msg=True)

#     def clear_vm_cache(self):
#         success: bool = self.executor.clear_vm_cache()
#         if success:
#             print_log(f"[PostgresqlDB] VM cache cleared.", print_msg=True)

#     def default_restart(self, exec: bool =False, dummy: bool = False):
#         self.executor.recover_default_config()
        
#         success, err_msg = self.knob_applier.kill_postgres()
#         if not success:
#             raise RuntimeError(f"[PostgresqlDB] Error in default restart (when terminating PG server) with error: {err_msg}")
#         success, err_msg = self.knob_applier.start_postgres()
#         if not success:
#             raise RuntimeError(f"[PostgresqlDB] Error in default restart (when launching PG server) with error: {err_msg}")

#         if exec:
#             self.step(config=None, dummy=dummy)

#     def step(self, config: Configuration, exec_overwrite: str = None, dummy: bool = False) -> dict:
#         optim_exec_time = perf_counter() - self.last_call_time
#         prep_time, bench_exec_time = 0, 0

#         prep_start_time = perf_counter()
#         self.executor.recover_default_config()
#         try:
#             if dummy:
#                 res = {self.args_db['perf_name']: random.random()}, []
#                 res['exec_success'] = True
#                 return res

            
#             if config:
#                 print_log("-" * 80)
#                 knobs = dict(config).copy()
#                 knob_applied, err_msg = self.knob_applier.apply_knobs(knobs, self.online_mode)
#                 print_log('-' * 80)
#             else:
#                 knob_applied, err_msg = True, None
#                 print_log(f'[PostgresqlDB] Default configuration applied.')
#             prep_time = perf_counter() - prep_start_time
            
#             bench_start_time = perf_counter()
#             if knob_applied:
#                 res = self.executor.run_benchmark(exec_overwrite=exec_overwrite)
#                 res['exec_success'] = True
#             else:
#                 res = {'exec_success': False, 'exec_error': err_msg}
#             bench_exec_time = perf_counter() - bench_start_time

#         except Exception as e:
#             print_log(f"[PostgresqlDB] Error in step: {e}\t{traceback.format_exc()}")
#             res = {'exec_success': False, 'exec_error': str(e)}
        
#         res['optim_exec_time'] = optim_exec_time
#         res['prep_time'] = prep_time
#         res['bench_exec_time'] = bench_exec_time

#         self.last_call_time = perf_counter()
#         return res
