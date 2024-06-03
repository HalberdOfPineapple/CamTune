import os
from dataclasses import dataclass
from camtune.utils import print_log, get_expr_name

# ----------------------------------------------------------------------------
RESTART_WAIT_TIME = 20
START_CONN_TRIALS = 30
TIMEOUT_CLOSE = 60

SINGLE_TIMEOUT_MAP = {
    'tpch_s': 50,
    'imdb': 100,
}
GROUP_TIMEOUT_MAP = {
    'tpch_s': 150,
    'imdb': 150,
}

EXEC_MODES = ['raw', 'pgbench', 'explain']
SINGLE_QUERY_TIMEOUT = 'SINGLE_QUERY_TIMEOUT'
ACCUM_QUERY_TIMEOUT = 'ACCUM_QUERY_TIMEOUT'

VACUUM_QUERY = "VACUUM FULL;"
ANALYZE_QUERY = "ANALYZE;"

# KILL_CMD_TEMP = '{} stop -m fast -D {}'
KILL_CMD_TEMP = '{} stop -D {}'
PGSQL_EXEC_TEMP = "psql -U {} -d {} -f {}"
PGSQL_EXEC_TEMP_RAW = "psql -U {} -d {} -c \"{}\""

PG_CHECK_CMD = "pgrep -u postgres -f -- -D"
RESTART_CMD = "systemctl restart postgresql"
RESTART_CMD_WSL = "service postgresql restart"

# Regular expression to extract the "real" component
REAL_PATTERN = r'real\s+(\d+)m([\d.]+)s'

# Define a custom exception for the timeout
class TimeoutException(Exception):
    pass

@dataclass
class DBConfig:
    on_wsl: bool = False
    db_host: str = "localhost"
    db_port: str = "5432"
    db_user: str = "wl446"
    db_pwd: str = "741286"
    db_name: str = "tpch"

    pg_ctl: str = "/usr/lib/postgresql/16/bin/pg_ctl"
    pg_data: str = "/home/wl446/pg_data/postgresql/16/main"
    pg_server: str = "/usr/lib/postgresql/16/bin/postgres"
    pg_conf: str = "/home/wl446/pg_data/postgresql/16/tune_conf.conf"
    pg_default_conf: str = "/home/wl446/pg_data/postgresql/16/template_conf.conf"
    pg_sock: str = "/var/run/postgresql/.s.PGSQL.5432"

    sys_user: str = "wl446"
    sys_pwd: str = "CL741286%"

    def __init__(self, args_db: dict, on_wsl: bool = False):
        self.on_wsl = on_wsl

        # Note that if using remote mode, all the paths here are remote paths
        self.pg_ctl = args_db.get('pg_ctl', self.pg_ctl)
        self.pg_data = args_db.get('pg_data', self.pg_data)
        self.pg_server = args_db.get('pg_server', self.pg_server)
        self.pg_conf = args_db.get('pg_conf', self.pg_conf)
        self.pg_default_conf = args_db.get('pg_default_conf', self.pg_default_conf)
        self.pg_sock = args_db.get('pg_sock', self.pg_sock)

        self.db_host = args_db.get('db_host', self.db_host)
        self.db_port = args_db.get('db_port', self.db_port)
        self.db_user = args_db.get('db_user_name', self.db_user)
        self.db_pwd = args_db.get('db_passwd', self.db_pwd)

        self.sys_user = args_db.get('sys_user', self.sys_user)
        self.sys_pwd = args_db.get('sys_pwd', self.sys_pwd)

        # ssh_key and ssh_passphrase are locally set for connecting to SSH server
        self.ssh_key = args_db.get('ssh_key', None)
        self.ssh_passphrase = args_db.get('ssh_passphrase', None)
        
        if 'benchmark' in args_db:
            self.init_db_name(args_db)

    @property
    def remote_tmp_dir(self):
        return os.path.join('/home', self.sys_user, 'tmp')
    
    def init_db_name(self, args_db: dict):
        benchmark = args_db['benchmark'].upper()
        scale_factor = args_db.get('scale_factor', None)

        if benchmark == 'TPCH':
            if scale_factor is None:
                self.db_name = "tpch"
            elif scale_factor == 2.5:
                self.db_name = "tpch_s"
            else:
                raise ValueError(f"Scale factor {scale_factor} is not supported for TPCH.")
        elif benchmark == 'SYSBENCH':
            sysbench_mode = args_db.get('sysbench_mode', 'read_write').lower()
            self.db_name = 'sysbench_test' if 'test' in sysbench_mode else 'sysbench_20G'
        elif 'YCSB' in benchmark:
            self.db_name = args_db.get('db_name', 'ycsb') if 'TEST' not in benchmark else 'ycsb_test'
        elif 'TPCC' in benchmark:
            self.db_name = args_db.get('db_name', 'tpcc') if 'TEST' not in benchmark else 'tpcc_test'
        elif benchmark == 'JOB':
            self.db_name = args_db.get('db_name', 'imdb')
        else:
            raise ValueError(f"Benchmark {benchmark} is not supported.")
    
    # Singleton instance storage
    _instance = None
    def build_instance(args_db: dict, on_wsl: bool = False):
        if DBConfig._instance is None:
            # if the instance is not built yet, we build it
            DBConfig._instance = DBConfig(args_db, on_wsl)
        else:
            # if the instance is already built, we directly modify it
            DBConfig._instance.__init__(args_db, on_wsl)
        return DBConfig._instance
    
    def get_instance():
        if DBConfig._instance is None:
            DBConfig._instance = DBConfig({})
        return DBConfig._instance
    
    def __str__(self):
        res_str: str = "DBConfig:\n"

        res_str += f"\ton_wsl: {self.on_wsl}\n"
        res_str += f"\tdb_host: {self.db_host}\n"
        res_str += f"\tdb_port: {self.db_port}\n"
        res_str += f"\tdb_user: {self.db_user}\n"
        
        res_str += f"\tdb_name: {self.db_name}\n"
        res_str += f"\tpg_ctl: {self.pg_ctl}\n"
        res_str += f"\tpg_data: {self.pg_data}\n"
        res_str += f"\tpg_server: {self.pg_server}\n"
        res_str += f"\tpg_conf: {self.pg_conf}\n"
        res_str += f"\tpg_default_conf: {self.pg_default_conf}\n"
        res_str += f"\tpg_sock: {self.pg_sock}\n"
        res_str += f"\tsys_user: {self.sys_user}\n"
        
        return res_str

def init_global_vars(args_db: dict, on_wsl: bool = False):
    return DBConfig.build_instance(args_db, on_wsl)

def get_db_config() -> DBConfig:
    return DBConfig.get_instance()

def modify_db_config(field: str, value):
    db_config: DBConfig = get_db_config()
    setattr(db_config, field, value)


def get_single_timeout(db_name: str):
    return SINGLE_TIMEOUT_MAP.get(db_name, 300)

def get_group_timeout(db_name: str):
    return GROUP_TIMEOUT_MAP.get(db_name, 300)
