"""
example cmdline:

python test/resnet/benchmark_resnet.py --mth hyperband --R 27 --n_workers 4 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from mq_hb import mth_dict
from benchmark_resnet_utils import run_exp

parser = argparse.ArgumentParser()
parser.add_argument('--mth', type=str, default='hyperband')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_jobs', type=int, default=4)

parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--n_workers', type=int)        # must set

parser.add_argument('--runtime_limit', type=int, default=86400)
parser.add_argument('--time_limit_per_trial', type=int, default=999999)

parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
algo_name = args.mth
dataset = args.dataset
R = args.R
eta = args.eta
n_jobs = args.n_jobs                                # changed according to dataset

ip = args.ip
port = args.port
n_workers = args.n_workers  # Caution: must set for saving result to different dirs

runtime_limit = args.runtime_limit                  # changed according to dataset
time_limit_per_trial = args.time_limit_per_trial    # changed according to dataset

rep = args.rep
start_id = args.start_id

print(ip, port, n_jobs, n_workers, dataset)
print(R, eta)
print(runtime_limit)
for para in (ip, port, n_jobs, R, eta, n_workers, runtime_limit):
    assert para is not None

algo_class, parallel_strategy = mth_dict[algo_name]

# objective_func, config_space, random_state, method_id, runtime_limit, time_limit_per_trial, ip, port
# are filled in run_exp()
algo_kwargs = dict(
    R=R, eta=eta,
    restart_needed=True,
)

run_exp(dataset, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
        R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
        eta=eta, pre_sample=False, run_test=True)
