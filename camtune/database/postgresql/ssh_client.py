import subprocess
import re
import socket
import time
import paramiko
import os
from subprocess import CompletedProcess, PIPE, Popen
from typing import Tuple, List

from camtune.utils import print_log, TEMP_DIR
from camtune.database.utils import run_as_user

from .variables import *
from .connector import PostgresqlConnector

DBCONFIG = get_db_config()
MODULE_NAME = "PGSSHClient"
PRINT_CHAR_LIMIT = 5000

def parse_timing_result(timing_result: str) -> Tuple[int, float]:
    # Search for the pattern in the timing result
    match = re.search(REAL_PATTERN, timing_result)

    if match:
        minutes = int(match.group(1))  # Extract the minutes part
        seconds = float(match.group(2))  # Extract the seconds part

        # Calculate the total time in seconds
        total_time_in_seconds = minutes * 60 + seconds
        return total_time_in_seconds
    else:
        return -1

START_OUT_PATH = os.path.join(TEMP_DIR, 'pg_start_out.log')
START_ERR_PATH = os.path.join(TEMP_DIR, 'pg_start_err.log')

# START_CMD_TEMP = '{} --config_file={} -D {} > /dev/null 2>{} &'
START_CMD_TEMP = '{} --config_file={} -D {} > {} 2>{} &'

# ERROR_CHECK_CMD = f'grep "ready to accept connections" {START_ERR_PATH} || grep "FATAL" {START_ERR_PATH}'
ERROR_CHECK_CMD = f'cat {START_ERR_PATH}'
OUT_CHECK_CMD = f'cat {START_OUT_PATH}'

KILL_RETRY_TIMES = 3

class PGSSHClient:
    def __init__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            hostname=DBCONFIG.db_host, 
            username=DBCONFIG.sys_user,
            key_filename=DBCONFIG.ssh_key,
            passphrase=DBCONFIG.ssh_passphrase,
        )
    
    def __del__(self):
        self.ssh.close()
    
    # ------------------------------------------------------------------
    # Basic Operations
    def remote_exec_command(
            self, 
            command: str, 
            sudo: bool = False, 
            as_postgre: bool = False, 
            timing: bool = False, 
            timeout: float=None
    ) -> Tuple[int, str, str, str]:
        # if timeout duration is set to zero, the associated timeout is disabled.
        timeout = timeout if timeout is not None else 0
        if as_postgre:
            command = f'echo "{DBCONFIG.sys_pwd}" | sudo -S -u postgres timeout {timeout} {command}'
        elif sudo:
            command = f'echo "{DBCONFIG.sys_pwd}" | sudo -S timeout {timeout} {command}'
        else:
            command = f'timeout {timeout} {command}'

        if timing:
            command = f'time {command}'

        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(command)
        return ssh_stdout.channel.recv_exit_status(), ssh_stdout.read().decode(), \
                ssh_stderr.read().decode(), command
    
    def remote_exec_sql(
            self,
            sql: str, 
            db_name: str=None, 
            db_user: str=None, 
            timing: bool=False, 
            timeout: float=None
    ):
        db_name = db_name if db_name else DBCONFIG.db_name
        db_user = db_user if db_user else DBCONFIG.db_user
        return self.remote_exec_command(
            PGSQL_EXEC_TEMP_RAW.format(db_user, db_name, sql),
            timing=timing, timeout=timeout
        )
    
    def remote_exec_sql_from_local_file(
            self,
            file_path: str,
            db_name: str=None, 
            db_user: str=None, 
            timing: bool=False, 
            timeout: float=None
    ):
        db_name = db_name if db_name else DBCONFIG.db_name
        db_user = db_user if db_user else DBCONFIG.db_user
        with open(file_path, 'r') as f:
            sql = f.read()

        return self.remote_exec_sql(
            sql, db_name=db_name, db_user=db_user, 
            timing=timing, timeout=timeout
        )

    def download_file(self, remote_path: str, local_path: str):
        sftp = self.ssh.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
    
    def upload_file(self, local_path: str, remote_path: str):
        sftp = self.ssh.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()

    def sync_file(self, local_path: str, remote_path: str):
        # Ensure the remote directory exists
        remote_dir = os.path.dirname(remote_path)
        mkdir_command = f'mkdir -p {remote_dir}'
        retcode, stdout, stderr, cmd = self.remote_exec_command(mkdir_command, sudo=True)  # Assuming sudo is required
        if retcode != 0:
            raise RuntimeError(f'[PGSSHClient-sync] Failed to create remote directory ({remote_dir}): {stderr}')

        # Upload the file
        self.upload_file(local_path, remote_path)

        # Match the local file permissions on the remote file
        local_perms = oct(os.stat(local_path).st_mode)[-3:]
        chmod_command = f'chmod {local_perms} {remote_path}'
        retcode, stdout, stderr, cmd = self.remote_exec_command(chmod_command, sudo=True) 
        if retcode != 0:
            raise RuntimeError(f'[PGSSHClient-sync] Failed to match permissions: {stderr}')
    
    def list_dir(self, dir: str) -> List[str]:
        """
        List the contents of a remote directory.
        """
        command = f'ls -A1 {dir}'  # -A1 lists all entries vertically except for . and ..
        exit_status, stdout, stderr, executed_command = self.remote_exec_command(command)

        if exit_status == 0:
            # Splitting stdout into a list, stripping out newlines
            return stdout.splitlines()
        else:
            # Handling errors such as the directory not existing
            raise Exception(f"[PGSSHClient-list-dir] Failed to list directory: {stderr}")

    def sync_from_remote_dir(self, local_dir: str, remote_dir: str):
        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List the contents of the remote directory
        remote_files = self.list_dir(remote_dir)
        for remote_file in remote_files:
            local_file = os.path.join(local_dir, str(remote_file))
            remote_file = os.path.join(remote_dir, str(remote_file))
            self.download_file(remote_file, local_file)
        
        print_log(f"[{MODULE_NAME}] Synced files from remote directory: {remote_dir} to local directory: {local_dir} successfully")
    
    def reboot(self):
        ret_code, stdout, stderr, cmd = self.remote_exec_command("reboot", sudo=True)
        if ret_code == 0:
            print_log(f"[{MODULE_NAME}] Remote host rebooted successfully", print_msg=True)
        else:
            print_log(f"[{MODULE_NAME}] Failed to reboot remote host with output: {stdout[:PRINT_CHAR_LIMIT]} and error info: {stderr[:PRINT_CHAR_LIMIT]}", print_msg=True)

    
    # ------------------------------------------------------------------
    # PostgreSQL Operations
    def check_pg_running(self) -> bool:
        print_log(f"[{MODULE_NAME}] Checking if PostgreSQL server is running with command: {PG_CHECK_CMD}...")
        ret_code, stdout, stderr, cmd = self.remote_exec_command(PG_CHECK_CMD)
        return ret_code == 0

    def kill_postgres(self) -> Tuple[bool, str]:
        kill_cmd = KILL_CMD_TEMP.format(DBCONFIG.pg_ctl, DBCONFIG.pg_data)

        # kill_trial_cnt = 0
        # while kill_trial_cnt < KILL_RETRY_TIMES:
        #     ret_code, ssh_stdout, ssh_stderr, cmd = self.remote_exec_command(kill_cmd, as_postgre=True)
        #     if ret_code == 0: break
        #     kill_trial_cnt += 1
        
        ret_code, ssh_stdout, ssh_stderr, cmd = self.remote_exec_command(kill_cmd, as_postgre=True)
        print_log(f"[{MODULE_NAME}] Kill PostgreSQL server with command: {cmd}")
        

        time.sleep(5)

        if ret_code == 0:
            print_log(f"[{MODULE_NAME}] Remote PostgreSQL server shut down successfully with output: {ssh_stdout[:PRINT_CHAR_LIMIT]}")
        elif 'Is server running' in ssh_stderr:
            print_log(f"[{MODULE_NAME}] Failed to shut down PostgreSQL server because the server has already been shut down ({ssh_stderr})")
        else:
            print_log(f"[{MODULE_NAME}] Failed to shut down PostgreSQL server with output: {ssh_stdout[:PRINT_CHAR_LIMIT]} and err: {ssh_stderr[:PRINT_CHAR_LIMIT]}")
            return False, ssh_stderr

        return True, None

    def start_postgres(self) -> Tuple[bool, str]:
        launch_cmd = START_CMD_TEMP.format(DBCONFIG.pg_server, DBCONFIG.pg_conf, DBCONFIG.pg_data, START_OUT_PATH, START_ERR_PATH)
        ret_code, std_out, std_err, cmd = self.remote_exec_command(launch_cmd, as_postgre=True)
        print_log(f"[{MODULE_NAME}] Launch PostgreSQL server with command: {cmd}")
        if ret_code != 0:
            print_log(f"[{MODULE_NAME}] Failed to execute PostgreSQL server launch command with output: {std_out[:PRINT_CHAR_LIMIT]} and err: {std_err[:PRINT_CHAR_LIMIT]}")
            return False
        
        
        count = 0
        print_log(f'[{MODULE_NAME}] Wait for connection to the started server...')
        while True:
            # Check error log for startup confirmation or errors
            ret_code, read_out_res, read_out_err, cmd = self.remote_exec_command(OUT_CHECK_CMD)
            ret_code, read_err_res, read_err_err, cmd = self.remote_exec_command(ERROR_CHECK_CMD)
            
            if 'ready to accept connections' in read_err_res or 'ready to accept connections' in read_out_res:
                print_log(f"[{MODULE_NAME}] Remote PostgreSQL server started successfully after {count+1} tries")
                return True, None
            elif 'FATAL' in read_err_res or 'FATAL' in read_out_res:
                print_log(f"[{MODULE_NAME}] Failed to start PostgreSQL server with output: {read_out_res[:PRINT_CHAR_LIMIT]} and err: {read_err_res[:PRINT_CHAR_LIMIT]}")
                return False, read_err_res

            time.sleep(1)
            count = count + 1
            if count >= START_CONN_TRIALS: 
                print_log(f"[{MODULE_NAME}] Failed to connect to newly-started PG server after {count} tries")
                break

        print_log(f"[{MODULE_NAME}] Failed to start PostgreSQL server with output: {read_out_res[:PRINT_CHAR_LIMIT]} and err: {read_err_res[:PRINT_CHAR_LIMIT]}")
        return False, "PostgrelSQL server launch timeout"
        
    

    # ------------------------------------------------------------------
    # Knob Applying
    def set_knob_value(self, knob: str, value: str) -> bool:
        if str(value).isdigit():
            sql = "SET {}={}".format(knob, value)
        else:
            sql = "SET {}='{}'".format(knob, value)

        retcode, ssh_stdout, ssh_stderr, cmd = self.remote_exec_sql(sql)
        if retcode != 0:
            print_log(f"[{MODULE_NAME}] Failed to set knob {knob} with value {value} with error info: {ssh_stderr[:PRINT_CHAR_LIMIT]}")

    def get_knob_value(self, knob: str) -> str:
        sql = 'SHOW {};'.format(knob)
        retcode, ssh_stdout, ssh_stderr, cmd = self.remote_exec_sql(sql)
        if retcode != 0:
            raise Exception(f"{ssh_stderr}")
        
        return ssh_stdout.strip('\n').split('\n')[-2].strip()

    