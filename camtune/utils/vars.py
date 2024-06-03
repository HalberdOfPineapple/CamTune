import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.double

OLTP_BENCHMARKS = {'sysbench', 'ycsba', 'ycsbb', 'tpcc'}
OLTP_BENCHMARKS = OLTP_BENCHMARKS.union({f'{benchmark}_test' for benchmark in OLTP_BENCHMARKS})
OLAP_BENCHMARKS = {'tpch', 'job'}
BENCHMARKS = OLTP_BENCHMARKS.union(OLAP_BENCHMARKS)
