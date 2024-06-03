import os
import re
import json
import subprocess

from camtune.utils import BENCHMARK_DIR

# ----------------------------------------------------------------------------------
# BenchBase
BENCHBASE_SCRIPT = os.path.join(BENCHMARK_DIR, "run_benchbase.sh")
BENCHBASE_LOAD_SCRIPT = os.path.join(BENCHMARK_DIR, "load_benchbase.sh")
BENCHBASE_CONFIG_DIR = os.path.join(BENCHMARK_DIR, "benchbase")
BENCHBASE_MODES = {'ycsba', 'ycsbb', 'ycsba_test', 'ycsbb_test', 'tpcc', 'tpcc_test'}
BENCHBASE_METRICS = ['throughput', 'latency']
BENCHBASE_TIMEOUT = 200
def get_benchmark_name(benchmark: str):
    if 'ycsb' in benchmark:
        return 'ycsb'
    elif 'tpcc' in benchmark:
        return 'tpcc'
    else:
        raise ValueError(f"[Workload-Benchbase] Benchmark {benchmark} is not supported.")
def parse_benchbase(res_dir: str):
    # find the file ending with `.summary.json` and load it
    try:
        summary_file = [file for file in os.listdir(res_dir) if file.endswith('.summary.json')][0]
    except Exception as e:
        raise FileNotFoundError(f"[Workload-Benchbase] Cannot find the summary file in {res_dir} ({e})")
    
    with open(os.path.join(res_dir, summary_file), 'r') as f:
        summary_dict: dict = json.load(f)
    
    res_dict = {
        'throughput': float(summary_dict['Throughput (requests/second)']),
        'latency': float(summary_dict['Latency Distribution']['95th Percentile Latency (microseconds)']) / 1000,
    }
    return res_dict

# ----------------------------------------------------------------------------------
# Sysbench
SYSBENCH_SCRIPTS_DIR = "/usr/local/share/sysbench/"
SYSBENCH_EXEC_TIME = 100 # 150 # 180
SYSBENCH_TEST_TIME = 10
SYSBENCH_CMD_TMP = (
     "sysbench {script} --db-driver=pgsql "
     "--pgsql-db={db_name} --pgsql-user={db_user} --pgsql-password='{db_pwd}' --pgsql-host={db_host} "
     "--tables={tables} --table-size={table_size} --threads=32 "
     "--range-size=100 --events=0 --rand-type=uniform --db-ps-mode=disable "
     "--time={time} run"
)
SYSBENCH_WORKLOADS = {
    'read_only': {
        'script': 'oltp_read_only.lua',
        'tables': 150,
        'table_size': 800000,
        'time': SYSBENCH_EXEC_TIME
    }, 
    'write_only': {
        'script': 'oltp_write_only.lua',
        'tables': 150,
        'table_size': 800000,
        'time': SYSBENCH_EXEC_TIME
    },
    'read_write_ratio_13': {
        'script': 'oltp_read_write.lua',
        'tables': 150,
        'table_size': 800000,
        'time': SYSBENCH_EXEC_TIME,
        'postfix': "--point_selects=11 --index_updates=11 --non_index_updates=12 --delete_inserts=11",
    },
    'read_write_20G': {
        'script': 'oltp_read_write.lua',
        'tables': 150,
        'table_size': 800000,
        'time': SYSBENCH_EXEC_TIME
    },
    'read_only_test': {
        'script': 'oltp_read_only.lua',
        'tables': 10,
        'table_size': 2000,
        'time': SYSBENCH_TEST_TIME,
    },
    'write_only_test': {
        'script': 'oltp_write_only.lua',
        'tables': 10,
        'table_size': 2000,
        'time': SYSBENCH_TEST_TIME,
    },
    'read_write_test': {
        'script': 'oltp_read_write.lua',
        'tables': 10,
        'table_size': 2000,
        'time': SYSBENCH_TEST_TIME,
    },
    'read_write_test_ratio_13': {
        'script': 'oltp_read_write.lua',
        'tables': 10,
        'table_size': 2000,
        'time': SYSBENCH_EXEC_TIME,
        'postfix': "--point_selects=11 --index_updates=11 --non_index_updates=12 --delete_inserts=11",
    },
}


def parse_sysbench_output(stats_output: str):
    lines = stats_output.split('\n')
    throughput, latency = None, None
    for line in lines:
        if '95th percentile:' in line:
            latency = float(line.strip().split()[-1])
        elif 'events/s (eps):' in line:
            throughput = float(line.strip().split()[-1])
    if throughput is None or latency is None:
        raise ValueError(f"[Workload-SysBench] Throughput or latency not found in the output: {stats_output}")
    res_dict = {
        'throughput': throughput,
        'latency': latency
    }
    return res_dict

# ----------------------------------------------------------------------------------
# YCSB
YCSB_DIR = "/home/wl446/YCSB/"
YCSB_TIME, YCSB_TEST_TIME = 80, 10
YCSB_BIN = os.path.join(YCSB_DIR, "bin/ycsb.sh")
YCSB_WORKLOAD_DIR = os.path.join(YCSB_DIR, "workloads")
YCSB_PROP = os.path.join(YCSB_DIR, "postgrenosql", "db.properties")

YCSB_WORKLOADS = {
    'ycsba': {
        'workload': 'workloada',
        'db_name': 'ycsba',
        'time': YCSB_TIME,
        'record_count': 15000000,
        'prop_path': os.path.join(YCSB_DIR, 'postgrenosql/a.properties')
    },
    'ycsbb': {
        'workload': 'workloadb',
        'db_name': 'ycsbb',
        'time': YCSB_TIME,
        'record_count': 15000000,
        'prop_path': os.path.join(YCSB_DIR, 'postgrenosql/b.properties')
    },
    'ycsba_test': {
        'workload': 'workloada',
        'db_name': 'ycsba_test',
        'time': YCSB_TEST_TIME,
        'record_count': 10000,
        'prop_path': os.path.join(YCSB_DIR, 'postgrenosql/a_test.properties')
    },
    'ycsbb_test': {
        'workload': 'workloadb',
        'db_name': 'ycsbb_test',
        'time': YCSB_TEST_TIME,
        'record_count': 10000,
        'prop_path': os.path.join(YCSB_DIR, 'postgrenosql/b_test.properties')
    },
}

def parse_ycsb_output(stats_output: str):
    stats_dict = {}

    for line in stats_output.split('\n'):
        if not line.startswith('['):
            continue

        aspect, field, val = line.split(',')
        if field.strip() == 'Throughput(ops/sec)':
            stats_dict = {'throughput': float(val.strip())}
    
    return stats_dict


