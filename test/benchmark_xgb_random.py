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
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import Process, Manager

sys.path.append(".")
sys.path.insert(0, "../lite-bo")  # for dependency
from mq_hb.mq_random_search import mqRandomSearch
from mq_hb.mq_mf_worker import mqmfWorker
from mq_hb.xgb_model import XGBoost
from utils import load_data, setup_exp, check_datasets, seeds

# default_datasets = 'mnist_784,higgs,covertype'
default_datasets = 'covtype,codrna'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=default_datasets)
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


def evaluate_parallel(method_id, n_workers, dataset, seed, ip, port):
    print(method_id, n_workers, dataset, seed)
    if port == 0:
        port = 13579 + np.random.randint(1000)
    print('ip=', ip, 'port=', port)

    def mf_objective_func(config, n_resource, extra_conf):
        print('objective extra conf:', extra_conf)
        params = config.get_dictionary()
        model = XGBoost(**params, n_jobs=n_jobs, seed=seed)
        model.fit(x_train, y_train)

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

    def master_run(return_list):
        rs = mqRandomSearch(None, cs, R, n_workers=n_workers,
                            num_iter=num_iter, random_state=seed,
                            method_id=method_id, restart_needed=True,
                            time_limit_per_trial=time_limit_per_trial,
                            runtime_limit=runtime_limit,
                            ip='', port=port)
        rs.run()
        return_list.extend(rs.recorder)  # send to return list

    def worker_run(i):
        worker = mqmfWorker(mf_objective_func, ip, port)
        worker.run()
        print("Worker %d exit." % (i,))

    manager = Manager()
    recorder = manager.list()  # shared list
    master = Process(target=master_run, args=(recorder,))
    master.start()

    time.sleep(10)  # wait for master init
    worker_pool = []
    for i in range(n_workers):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()  # wait for master to gen result
    for w in worker_pool:  # optional if repeat=1
        w.join()

    return list(recorder)  # covert to list


check_datasets(test_datasets)
for dataset in test_datasets:
    # setup
    n_jobs, runtime_limit, time_limit_per_trial = setup_exp(dataset, n_jobs, runtime_limit, time_limit_per_trial)
    print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
          % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        method_str = 'random-n%d' % (n_workers,)
        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

        recorder = evaluate_parallel(method_id, n_workers, dataset, seed, ip, port)

        dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, runtime_limit, method_str)
        file_name = 'record_%s.pkl' % (method_id,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            pkl.dump(recorder, f)
        print(dir_path, file_name, 'saved!', flush=True)
