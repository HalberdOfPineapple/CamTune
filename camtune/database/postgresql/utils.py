import re
import time
import psycopg2
import traceback
from subprocess import CompletedProcess, PIPE, Popen
from typing import Tuple

from camtune.utils import print_log
from camtune.database.utils import run_as_user

from .variables import *

DBCONFIG = get_db_config()
START_CMD_TEMP = '{} --config_file={} -D {}'

def run_command(
        command:str, 
        sudo: bool=False, 
        as_postgres: bool=False,
        timeout: int=None,
    ) -> Tuple[CompletedProcess, str, str, str]:
    if as_postgres:
        return run_as_user(command, password=DBCONFIG.sys_pwd, user='postgres', timeout=timeout)
    elif sudo:
        return run_as_user(command, password=DBCONFIG.sys_pwd, timeout=timeout)
    else:
        return run_as_user(command, timeout=timeout)

def run_sql_query(query: str, db_user: str=None, db_name: str = None, timeout:int = None) -> Tuple[str, str]:
    db_user = db_user if db_user else DBCONFIG.db_user
    db_name = db_name if db_name else DBCONFIG.db_name

    exec_cmd = PGSQL_EXEC_TEMP_RAW.format(db_user, db_name, query)
    proc, std_out, std_err, exec_cmd = run_command(exec_cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"[PGUtils] Failed to execute SQL query ({query[:50]}...): {std_err}")
    return std_out, std_err

def run_sql_from_file(file_path: str, db_user: str=None, db_name: str = None, timeout: int=None) -> Tuple[str, str]:
    db_user = db_user if db_user else DBCONFIG.db_user
    db_name = db_name if db_name else DBCONFIG.db_name

    exec_cmd = PGSQL_EXEC_TEMP.format(db_user, db_name, file_path)
    proc, std_out, std_err, exec_cmd = run_command(exec_cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"[PGUtils] Failed to execute SQL file ({file_path}): {std_err}")
    return std_out, std_err

def check_pg_running() -> bool:
    proc, stdout, stderr, cmd = run_command(PG_CHECK_CMD)
    return eval(stdout) > 0

# --------------------------------------------------------------------
def start_postgres():
    launch_cmd = START_CMD_TEMP.format(DBCONFIG.pg_server, DBCONFIG.pg_conf, DBCONFIG.pg_data)
    proc, std_out, std_err, launch_cmd = run_command(launch_cmd, as_postgres=True)
    print_log(f'[PGUtils] Locally starting PostgreSQL server with command: {launch_cmd} ...', print_msg=True)

    # return True
    return try_connect_pg()

def kill_postgres():
    kill_cmd = KILL_CMD_TEMP.format(DBCONFIG.pg_ctl, DBCONFIG.pg_data)
    proc, std_out, std_err, kill_cmd = run_command(kill_cmd, as_postgres=True, timeout=TIMEOUT_CLOSE)
    ret_code = proc.returncode

    if ret_code == 0:
        print_log("[PGUtils] Local PostgreSQL server shut down successfully", print_msg=True)
    else:
        print_log(f"[PGUtils] Local shut down attempt ({kill_cmd}) failed with error info: {std_err}", print_msg=True)
        return False, std_err
    return True, None

def try_connect_pg():
    count = 0
    start_success = True
    print_log('[PGUtils] Wait for connection to the started server...')
    while count < START_CONN_TRIALS:
        try:
            db_conn = psycopg2.connect(
                host=DBCONFIG.db_host,
                user=DBCONFIG.db_user,
                password=DBCONFIG.db_pwd,
                database=DBCONFIG.db_name,
                port=DBCONFIG.db_port,
            )
            if db_conn.closed == 0:
                print_log('[PGUtils] Successfully connected to the started PostgreSQL Server')
                db_conn.close()
                return True, None
        except Exception as e:
            if count % 5 == 0:
                # print_log(f'[PGUtils] Trial {count} to connect to the started PostgreSQL Server failed with error information {e} and traceback: {traceback.format_exc()}')
                print_log(f'[PGUtils] Trial {count} to connect to the started PostgreSQL Server failed with error information {e}')
            pass

        time.sleep(1)
        count = count + 1
    print_log(f"[PGUtils] Failed to connect to newly-started PG server after {count} tries")
    return False, str(e)
    


def measure_query_time(query_file_path: str) -> Tuple[str, float]:
    exec_cmd = PGSQL_EXEC_TEMP.format(DBCONFIG.db_user, DBCONFIG.db_name, query_file_path)

    start_time = time.perf_counter()
    proc, std_out, std_err, exec_cmd = run_command(exec_cmd, timeout=SINGLE_TIMEOUT_MAP[DBCONFIG.db_name])
    elapsed_time = time.perf_counter() - start_time

    return std_out, elapsed_time

# -------------------------------------------------------------------
# Results Processing

def extract_sql(query_for_time: str):
    start_token = "query := '"
    end_token = "';"

    start_index = query_for_time.find(start_token) + len(start_token)
    end_index = query_for_time.find(end_token, start_index)
    extracted_query = query_for_time[start_index:end_index]

    # Correcting for escaped single quotes within the SQL query
    extracted_query_corrected = extracted_query.replace("''", "'")

    return extracted_query_corrected

def extract_total_cost(plan):
    return float(plan.split('cost=')[1].split('..')[1].split(' ')[0])


def parse_pgbench_output(output: str):
    parsed_data = {}
    patterns = {
        'transaction_type': r'transaction type:\s*(.+)',
        'scaling_factor': r'scaling factor:\s*(.+)',
        'query_mode': r'query mode:\s*(.+)',
        'number_of_clients': r'number of clients:\s*(\d+)',
        'number_of_threads': r'number of threads:\s*(\d+)',
        'number_of_transactions': r'number of transactions actually processed:\s*(\d+)/\d+',
        'failed_transactions': r'number of failed transactions:\s*(\d+)',
        'latency_average': r'latency average = ([\d.]+)\s*ms',
        'initial_connection_time': r'initial connection time = ([\d.]+)\s*ms',
        'tps': r'tps = ([\d.]+)\s*'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            parsed_data[key] = match.group(1)

    return parsed_data