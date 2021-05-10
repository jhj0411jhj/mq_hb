"""
example cmdline:

python test/benchmark_xgb_async_mfes_v7.py --datasets spambase --R 27 --n_jobs 4 --n_workers 1 --rand_prob 0.3 \
--skip_outer_loop 0 --runtime_limit 60 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../lite-bo")    # for dependency
from mq_hb.async_mq_mfes_v7 import async_mqMFES_v7
from benchmark_xgb_utils import run_exp

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_jobs', type=int, default=4)

parser.add_argument('--skip_outer_loop', type=int, default=0)
parser.add_argument('--rand_prob', type=float, default=0.3)

parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--n_workers', type=int)        # must set

parser.add_argument('--runtime_limit', type=int, default=60)
parser.add_argument('--time_limit_per_trial', type=int, default=600)

parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))
R = args.R
eta = args.eta
n_jobs = args.n_jobs                                # changed according to dataset

skip_outer_loop = args.skip_outer_loop
rand_prob = args.rand_prob

ip = args.ip
port = args.port
n_workers = args.n_workers  # Caution: must set for saving result to different dirs

runtime_limit = args.runtime_limit                  # changed according to dataset
time_limit_per_trial = args.time_limit_per_trial    # changed according to dataset

rep = args.rep
start_id = args.start_id

print(ip, port, n_jobs, n_workers, test_datasets)
print(R, eta)
print(runtime_limit)
for para in (ip, port, n_jobs, R, eta, n_workers, runtime_limit):
    assert para is not None

algo_name = 'amfesv7'
algo_class = async_mqMFES_v7
# objective_func, config_space, random_state, method_id, runtime_limit, time_limit_per_trial, ip, port
# are filled in run_exp()
algo_kwargs = dict(
    R=R, eta=eta,
    skip_outer_loop=skip_outer_loop,
    rand_prob=rand_prob,
    restart_needed=True,
)
parallel_strategy = 'async'

run_exp(test_datasets, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
        R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
        eta=eta, pre_sample=False, run_test=True)
