"""
example cmdline:

python test/benchmark_xgb_random.py --datasets spambase --R 27 --n_jobs 4 --n_workers 1 \
--num_iter 10000 --runtime_limit 60 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")  # for dependency
from mq_hb.mq_random_search import mqRandomSearch
from benchmark_xgb_utils import run_exp

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--n_jobs', type=int, default=4)

parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--n_workers', type=int)  # must set

parser.add_argument('--num_iter', type=int, default=10000)
parser.add_argument('--runtime_limit', type=int, default=60)
parser.add_argument('--time_limit_per_trial', type=int, default=600)

parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))
R = args.R
n_jobs = args.n_jobs  # changed according to dataset

ip = args.ip
port = args.port
n_workers = args.n_workers  # Caution: must set for saving result to different dirs

num_iter = args.num_iter
runtime_limit = args.runtime_limit  # changed according to dataset
time_limit_per_trial = args.time_limit_per_trial  # changed according to dataset

rep = args.rep
start_id = args.start_id

print(ip, port, n_jobs, n_workers, test_datasets)
print(R)
print(num_iter, runtime_limit)
for para in (ip, port, n_jobs, R, n_workers, runtime_limit):
    assert para is not None

algo_name = 'random'
algo_class = mqRandomSearch
# objective_func, config_space, random_state, method_id, runtime_limit, time_limit_per_trial, ip, port
# are filled in run_exp()
algo_kwargs = dict(
    R=R,
    n_workers=n_workers,
    num_iter=num_iter,
    restart_needed=True,
)
parallel_strategy = 'sync'

run_exp(test_datasets, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
        R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
        eta=None, pre_sample=False, run_test=True)
