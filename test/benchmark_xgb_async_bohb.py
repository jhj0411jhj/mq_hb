"""
example cmdline:

python test/benchmark_xgb_async_bohb.py --datasets spambase --R 27 --n_jobs 4 --n_workers 1 --rand_prob 0.3 \
--skip_outer_loop 0 --runtime_limit 60 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import Process, Manager

sys.path.insert(0, ".")
sys.path.insert(1, "../lite-bo")    # for dependency
from mq_hb.async_mq_mfes import async_mqMFES    # use_bohb=True
from mq_hb.async_mq_mf_worker import async_mqmfWorker
from mq_hb.xgb_model import XGBoost
from utils import load_data, setup_exp, check_datasets, seeds

# default_datasets = 'mnist_784,higgs,covertype'
default_datasets = 'covtype,codrna'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=default_datasets)
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


def evaluate_parallel(method_id, n_workers, dataset, seed, ip, port):
    print(method_id, n_workers, dataset, seed)
    if port == 0:
        port = 13579 + np.random.randint(1000)
    print('ip=', ip, 'port=', port)

    def mf_objective_func(config, n_resource, extra_conf, total_resource, x_train, x_val, y_train, y_val, seed):
        print('objective extra conf:', extra_conf)
        params = config.get_dictionary()

        # sample train data. the test data after split is sampled train data
        if n_resource < total_resource:
            ratio = n_resource / total_resource
            print('sample data: ratio =', ratio, n_resource, total_resource)
            _x, sample_x, _y, sample_y = train_test_split(x_train, y_train, test_size=ratio,
                                                          stratify=y_train, random_state=seed)
        else:
            print('sample data: use full dataset', n_resource, total_resource)
            sample_x, sample_y = x_train, y_train

        model = XGBoost(**params, n_jobs=n_jobs, seed=seed)
        model.fit(sample_x, sample_y)

        # evaluate on validation data
        y_pred = model.predict(x_val)
        perf = -balanced_accuracy_score(y_val, y_pred)  # minimize

        result = dict(
            objective_value=perf,
            early_stop=False,  # for deep learning
            ref_id=None,
        )
        return result

    cs = XGBoost.get_cs()
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)
    mf_objective_func = partial(mf_objective_func, total_resource=R, seed=seed,
                                x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)

    def master_run(return_list):
        algo = async_mqMFES(
            None, cs, R, eta=eta,
            skip_outer_loop=skip_outer_loop,
            rand_prob=rand_prob,
            use_bohb=True,        # use BOHB
            random_state=seed,
            method_id=method_id, restart_needed=True,
            time_limit_per_trial=time_limit_per_trial,
            runtime_limit=runtime_limit,
            ip='', port=port,
            )
        algo.run()
        algo.logger.info('===== bracket status: %s' % algo.get_bracket_status(algo.bracket))
        return_list.extend(algo.recorder)  # send to return list

    def worker_run(i):
        worker = async_mqmfWorker(mf_objective_func, ip, port)
        worker.run()
        print("Worker %d exit." % (i,))

    manager = Manager()
    recorder = manager.list()   # shared list
    master = Process(target=master_run, args=(recorder,))
    master.start()

    time.sleep(10)  # wait for master init
    worker_pool = []
    for i in range(n_workers):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:   # optional if repeat=1
        w.join()

    return list(recorder)   # covert to list


check_datasets(test_datasets)
for dataset in test_datasets:
    # setup
    n_jobs, runtime_limit, time_limit_per_trial = setup_exp(dataset, n_jobs, runtime_limit, time_limit_per_trial)
    print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
          % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        method_str = 'abohb-n%d' % (n_workers,)
        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

        recorder = evaluate_parallel(method_id, n_workers, dataset, seed, ip, port)

        dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, runtime_limit, method_str)
        file_name = 'record_%s.pkl' % (method_id,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            pkl.dump(recorder, f)
        print(dir_path, file_name, 'saved!', flush=True)
