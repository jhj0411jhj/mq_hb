"""
example cmdline:

python test/benchmark_xgb_async_hb_v2.py --datasets spambase --R 27 --n_jobs 4 --n_workers 1 \
--skip_outer_loop 0 --runtime_limit 60 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from mq_hb.async_mq_hb_v0 import async_mqHyperband_v0

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_jobs', type=int, default=4)

parser.add_argument('--skip_outer_loop', type=int, default=0)

parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=4)        # must set

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


import os
import time
import traceback
import pickle as pkl
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import Process, Manager

from mq_hb.mq_mf_worker import mqmfWorker
from mq_hb.async_mq_mf_worker import async_mqmfWorker
from mq_hb.xgb_model import XGBoost
from utils import load_data, setup_exp, check_datasets, seeds


def mf_objective_func(config, n_resource, extra_conf,
                      total_resource,
                      n_jobs, run_test=True):
    np.random.seed(int(time.time()*10000%10000))
    perf = np.random.rand()
    time.sleep(0.1*n_resource)

    result = dict(
        objective_value=perf,
        early_stop=False,  # for deep learning
        ref_id=None,
        test_perf=None,
    )
    return result


def evaluate_parallel(algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
                      parallel_strategy, n_jobs, R, eta=None, pre_sample=False, run_test=True):
    print(method_id, n_workers, dataset, seed)
    if port == 0:
        port = 13579 + np.random.randint(1000)
    print('ip=', ip, 'port=', port)
    assert parallel_strategy in ['sync', 'async']
    if pre_sample and eta is None:
        raise ValueError('eta must not be None if pre_sample=True')

    if pre_sample:
        raise NotImplementedError   # todo
    else:
        objective_function = partial(mf_objective_func, total_resource=R,
                                     n_jobs=n_jobs, run_test=run_test)

    def master_run(return_list, algo_class, algo_kwargs):
        algo_kwargs['ip'] = ''
        algo_kwargs['port'] = port
        algo = algo_class(**algo_kwargs)
        algo.run()
        try:
            algo.logger.info('===== bracket status: %s' % algo.get_bracket_status(algo.bracket))
        except Exception as e:
            pass
        try:
            algo.logger.info('===== brackets status: %s' % algo.get_brackets_status(algo.brackets))
        except Exception as e:
            pass
        return_list.extend(algo.recorder)  # send to return list

    def worker_run(i):
        if parallel_strategy == 'sync':
            worker = mqmfWorker(objective_function, ip, port)
        elif parallel_strategy == 'async':
            worker = async_mqmfWorker(objective_function, ip, port)
        else:
            raise ValueError('Error parallel_strategy: %s.' % parallel_strategy)
        worker.run()
        print("Worker %d exit." % (i,))

    manager = Manager()
    recorder = manager.list()   # shared list
    master = Process(target=master_run, args=(recorder, algo_class, algo_kwargs))
    master.start()

    time.sleep(2)  # wait for master init
    worker_pool = []
    for i in range(n_workers):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:
        w.kill()

    return list(recorder)   # covert to list


def run_exp(test_datasets, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
            R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
            eta=None, pre_sample=False, run_test=True):
    for dataset in test_datasets:
        # setup
        n_jobs, runtime_limit, time_limit_per_trial = setup_exp(dataset, n_jobs, runtime_limit, time_limit_per_trial)
        print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
              % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
        for i in range(start_id, start_id + rep):
            seed = seeds[i]

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            method_str = '%s-n%d' % (algo_name, n_workers)
            method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

            # ip, port are filled in evaluate_parallel()
            # def get_cs():  # test small space
            #     from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter
            #     cs = ConfigurationSpace()
            #     max_depth = UniformIntegerHyperparameter("max_depth", 1, 1000)
            #     cs.add_hyperparameters([max_depth])
            #     return cs
            algo_kwargs['objective_func'] = None
            algo_kwargs['config_space'] = XGBoost.get_cs()
            algo_kwargs['random_state'] = seed
            algo_kwargs['method_id'] = method_id
            algo_kwargs['runtime_limit'] = runtime_limit
            algo_kwargs['time_limit_per_trial'] = time_limit_per_trial

            recorder = evaluate_parallel(
                algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
                parallel_strategy, n_jobs, R, eta=eta, pre_sample=pre_sample, run_test=run_test,
            )


algo_name = 'ahbv0'
algo_class = async_mqHyperband_v0
# objective_func, config_space, random_state, method_id, runtime_limit, time_limit_per_trial, ip, port
# are filled in run_exp()
algo_kwargs = dict(
    R=R, eta=eta,
    skip_outer_loop=skip_outer_loop,
    restart_needed=True,
)
parallel_strategy = 'async'

run_exp(test_datasets, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
        R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
        eta=eta, pre_sample=False, run_test=True)
