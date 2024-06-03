import re
import os
import json
import time
import socket
import paramiko
import traceback
import subprocess
from typing import List, Tuple, Optional, Union

from camtune.utils import print_log, get_logger
from camtune.utils import BENCHMARK_DIR, get_benchmark_name, TEMP_DIR
from camtune.database.config_parser import ConfigParser

from .connector import PostgresqlConnector
from .variables import *
from .utils import check_pg_running, kill_postgres, start_postgres, run_command, measure_query_time
from .ssh_client import PGSSHClient

DBCONFIG = get_db_config()
FACTOR_MAP = {
    'us': 1 / 1000000,
    'ms': 1 / 1000,
    's': 1,
    'min': 60,
    'h': 3600,
    'd': 3600 * 24,

    'b': 1 / 1024,
    'kb': 1,
    'mb': 1024,
    'gb': 1024 * 1024,
}

def parse_actual_value(actual_value):
    # Define regular expression patterns to match numeric value and unit
    value_pattern = r'(-?\d+(\.\d+)?)'  # Matches one or more digits, optionally followed by a decimal point and more digits
    unit_pattern = r'([a-zA-Z]+)'      # Matches one or more alphabetic characters (units)

    # Compile regular expression patterns
    value_regex = re.compile(value_pattern)
    unit_regex = re.compile(unit_pattern)

    # Find numeric value and unit in the actual value string
    value_match = value_regex.search(actual_value)
    unit_match = unit_regex.search(actual_value)

    # Extract numeric value and unit if found
    numeric_value = float(value_match.group(1)) if value_match else None
    unit = unit_match.group(1) if unit_match else None

    return numeric_value, unit


def get_expected_factor(unit: str, byte_unit: str):
    # sample: unit = 'kb, byte_unit = 1
    if unit is None: unit = 'kb'
    if byte_unit is None: byte_unit = 1

    factor = FACTOR_MAP[unit.lower()]
    return factor * int(byte_unit)

def get_actual_factor(unit: str):
    if unit is None: return 1
    return FACTOR_MAP[unit.lower()]
    

class PostgreKnobApplier:
    def __init__(self, remote_mode: bool, knob_details: dict, ssh_cli: PGSSHClient = None):
        self.knob_details = knob_details
        self.remote_mode = remote_mode

        self.ssh_cli = ssh_cli if self.remote_mode else None
        self.db_conn = PostgresqlConnector() if not self.remote_mode else None
    
    def warmup(self):
        warmup_file_path = os.path.join(BENCHMARK_DIR, f'{get_benchmark_name().lower()}_warmup.sql')
        _, warmup_time = measure_query_time(warmup_file_path)
        print_log(f'[PGKnobApplier] Warmup on executing {warmup_file_path} takes {warmup_time}s')
        return warmup_time
    
    def check_pg_running(self):
        if self.remote_mode:
            return self.ssh_cli.check_pg_running()
        else:
            return check_pg_running()

    def kill_postgres(self) -> Tuple[bool, Optional[str]]:
        if self.remote_mode:
            return self.ssh_cli.kill_postgres()
        else:
            return kill_postgres()
    
    def start_postgres(self) -> Tuple[bool, Optional[str]]:
        if self.remote_mode:
            return self.ssh_cli.start_postgres()
        else:
            return start_postgres()
    
    def apply_knobs(self, knobs: dict, online: bool):
        print_log(f"[PGKnobApplier] Applying knobs {'online' if online else 'offline'}: {list(knobs.keys())}")
        if online:
            return self.apply_knobs_online(knobs)
        else:
            return self.apply_knobs_offline(knobs)
    
    def adjust_knob_value(self, knobs: dict) -> dict:
        if 'min_wal_size' in knobs.keys():
            if 'wal_segment_size' in knobs.keys():
                wal_segment_size = knobs['wal_segment_size']
            else:
                wal_segment_size = 16
            if knobs['min_wal_size'] < 2 * wal_segment_size:
                knobs['min_wal_size'] = 2 * wal_segment_size
                print_log('[PGKnobApplier] Knob "min_wal_size" must be at least twice "wal_segment_size"')
        
        if 'log_statement_stats' in knobs.keys():
            others_enable = knobs.get('log_executor_stats', False) or knobs.get('log_planner_stats', False) or knobs.get('log_parser_stats', False)
            if others_enable:
                knobs['log_statement_stats'] = 'off'

        return knobs

    def apply_knobs_offline(self, knobs: dict) -> Tuple[bool, Optional[str]]:
        success, err_msg = self.kill_postgres()
        if not success:
            # raise RuntimeError("[PGKnobApplier] PostgreSQL failed to shut down before applying knobs offline.")
            return False, err_msg

        # --------------------------------------------------------------------
        # Adjust knobs values by modifying the configuration file offline
        knobs = self.adjust_knob_value(knobs)
        knobs_not_in_cnf = self.modify_config_file(knobs)

        # --------------------------------------------------------------------
        # If PostgreSQL server cannot start normally, terminate the program
        success, err_msg = self.start_postgres()
        if not success:
            # raise RuntimeError("[PGKnobApplier] PostgreSQL failed to start after applying knobs offline.")
            return False, err_msg

        sleep_time = RESTART_WAIT_TIME if os.getenv('MY_TEST_ENV') is None else 5
        print_log('[PGKnobApplier] Sleeping for {} seconds after restarting postgres'.format(sleep_time))
        time.sleep(sleep_time)

        # --------------------------------------------------------------------
        # Apply knobs that have not been written in configuration file online
        # knobs_not_applied = self.check_knobs_applied(knobs, online=True)

        if len(knobs_not_in_cnf) > 0:
            tmp_rds = {}
            for knob_rds in knobs_not_in_cnf:
                tmp_rds[knob_rds] = knobs[knob_rds]
            self.apply_knobs_online(tmp_rds)
        else:
            print_log("[PGKnobApplier] No knobs need to be applied online")
        # self.check_knobs_applied(knobs, online=False)

        return True, None

    def apply_knobs_online(self, knobs: dict) -> Tuple[bool, Optional[str]]:
        # apply knobs remotely
        print_log(f"[PGKnobApplier] Knobs to be applied online: {list(knobs.keys())}")
        if self.remote_mode:
            for key in knobs.keys():
                self.ssh_cli.set_knob_value(key, knobs[key])
        else:
            for key in knobs.keys():
                self.db_conn.set_knob_value(key, knobs[key])

        self.check_knobs_applied(knobs, online=True)
        return True, None
    
    # --------------------------------------------------------------------
    # Knob checking
    def get_unit(self, knob_name: str) -> Tuple[str, str]:
        unit = None if 'unit' not in self.knob_details[knob_name] else self.knob_details[knob_name]['unit']
        byte_unit = None if 'byte_unit' not in self.knob_details[knob_name] else self.knob_details[knob_name]['byte_unit']
        return unit, byte_unit
    
    def check_single_knob_apply(self, knob_name: str, actual_val: str, expected_val: Union[int, float, str]) -> bool:
        if isinstance(expected_val, str):
            return actual_val.lower() == expected_val.lower()
        elif actual_val == '-1':
            return expected_val == -1
            
        expected_unit, expected_byte_unit = self.get_unit(knob_name)
        expected_factor = get_expected_factor(expected_unit, expected_byte_unit)

        actual_val, actual_unit = parse_actual_value(actual_val)
        actual_factor = get_actual_factor(actual_unit)

        return (expected_val * expected_factor - actual_val * actual_factor) < 1e-5

    def check_knobs_applied(self, knobs: dict, online: bool) -> int:
        knob_not_applied = []
        for knob, knob_val in knobs.items():
            actual_val = None
            try:
                if knob in self.knob_details:
                    if self.remote_mode:
                        actual_val = self.ssh_cli.get_knob_value(knob)
                    else:
                        actual_val = self.db_conn.get_knob_value(knob)
                        
                    applied = self.check_single_knob_apply(knob, actual_val, knob_val)
                    if not applied:
                        knob_not_applied.append(knob)
                        print_log(f"[PGKnobApplier] Knob {knob} is not successfully set to expeced value: {knob_val} (type={type(knob_val)}) (actual value: {actual_val}, type={type(actual_val)})")
            except Exception as e:
                print_log(f"[PGKnobApplier] Error occurred while checking knob {knob} (actual value: {actual_val}): error: {e} and traceback: {traceback.format_exc()}")

        check_mode = "online" if online else "offline"
        if len(knob_not_applied) > 0:
            print_log(f"[PGKnobApplier] {len(knob_not_applied)} / {len(knobs)} knobs not successfully applied {check_mode}.")
        elif len(knob_not_applied) == 0:
            print_log(f"[PGKnobApplier] Knobs successfully applied {check_mode}.")
        return knob_not_applied
    
    # --------------------------------------------------------------------
    # Configuration file modification
    def modify_config_file(self, knobs: dict):
        """
        Note there three different resources of knob information: 
            - knobs: the knobs to be applied, 
            - knob_details: the details of all knobs,
            - config_parser.knobs: the knobs read from configuration file. 
        knobs_not_in_cnf should include the knobs that are not in `knob_details` but in `knobs` 
        (config_parser can add the new knob even if it does not originally exist in the configuration file or commented out)
        """
        knobs_not_in_cnf = set([knob for knob in knobs if knob not in self.knob_details])
        knobs_to_apply = {knob: knobs[knob] for knob in knobs if knob not in knobs_not_in_cnf}
        if self.remote_mode: 
            self.modify_config_file_remote(knobs_to_apply)
        else:
            self.modify_config_file_local(knobs_to_apply)
        return knobs_not_in_cnf
            
    def modify_config_file_local(self, knobs_to_apply: dict):
        config_parser = ConfigParser(DBCONFIG.pg_conf)
        tmp_cnf_path = config_parser.write_to_tmp_file(knobs_to_apply, tmp_file_path='/tmp/pg_tmp.cnf')

        cp_cmd = 'cp {} {}'.format(tmp_cnf_path, DBCONFIG.pg_conf)
        proc, std_out, std_err, cmd = run_command(cp_cmd, as_postgres=True)
        if proc.returncode != 0:
            print_log(f"[PGKnobApplier] Local configuration file modification failed: {std_err}")
        else:
            print_log('[PGKnobApplier] config file modification done (locally).', print_msg=True)
        
    def modify_config_file_remote(self, knobs_to_apply: dict):
        remote_local_conf = os.path.join(DBCONFIG.remote_tmp_dir, 'pg_remote_tmp.cnf')
        local_conf = os.path.join(TEMP_DIR, 'pg_local.cnf')
        tmp_local_conf = os.path.join(TEMP_DIR, 'pg_local_tmp.cnf')

        try:
            remote_cp_cmd = f'cp {DBCONFIG.pg_conf} {remote_local_conf}'
            remote_access_grant = f'chmod 777 {remote_local_conf}'

            retcode, _, stderr, _ = self.ssh_cli.remote_exec_command(remote_cp_cmd, as_postgre=True)
            if retcode != 0: raise RuntimeError(stderr)

            retcode, _, stderr, _ = self.ssh_cli.remote_exec_command(remote_access_grant, sudo=True)
            if retcode != 0: raise RuntimeError(stderr)
            
            self.ssh_cli.download_file(remote_local_conf, local_conf)
        except Exception as e:
            print_log(f'[PGKnobApplier] Getting PostgreSQL configuration file failed with error: {e} and traceback: {traceback.format_exc()}')

        config_parser = ConfigParser(local_conf)
        tmp_local_conf = config_parser.write_to_tmp_file(knobs_to_apply, tmp_file_path=tmp_local_conf)

        try:
            self.ssh_cli.upload_file(tmp_local_conf, remote_local_conf)
            remote_cp_cmd = f'cp {remote_local_conf} {DBCONFIG.pg_conf}'
            self.ssh_cli.remote_exec_command(remote_cp_cmd, as_postgre=True)
        except Exception as e:
            print_log(f'[PGKnobApplier] Uploading PostgreSQL configuration file failed with error: {e} and traceback: {traceback.format_exc()}')

        print_log('[PGKnobApplier] config file modification done remotely.')


if __name__ == '__main__':
    import sys, json
    from camtune.utils import RESULT_DIR, KNOB_DIR
    from camtune.database.utils import initialize_knobs

    knob_definition_path = os.path.join(KNOB_DIR, 'llamatune_90.json')
    knob_details = initialize_knobs(knob_definition_path, -1)

    benchmark_name, expr_name, config_idx = sys.argv[1], sys.argv[2], sys.argv[3]
    result_path = os.path.join(RESULT_DIR, benchmark_name, f'{expr_name}_data.json')
    with open(result_path, 'r') as f:
        configs = json.load(f)
    config_dict = configs[config_idx]['config']  

    knob_applier = PostgreKnobApplier(remote_mode=False, knob_details=knob_details)
    knob_applier.modify_config_file_local(config_dict)


    
