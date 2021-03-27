"""
example cmdline:

python test/test_async_sh.py --R 27 --eta 3 --n_workers 1 --runtime_limit 60

"""

import argparse
import numpy as np
import time
from multiprocessing import Process, Manager

import sys
sys.path.append('.')
sys.path.insert(0, '../lite-bo')    # for dependency
from mq_hb.async_mq_sh import async_mqSuccessiveHalving
from mq_hb.async_mq_mf_worker import async_mqmfWorker

from litebo.utils.config_space import ConfigurationSpace, Configuration, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--runtime_limit', type=int, default=60)
parser.add_argument('--time_limit_per_trial', type=int, default=60)

args = parser.parse_args()
ip = args.ip
port = args.port
R = args.R
eta = args.eta
n_workers = args.n_workers
runtime_limit = args.runtime_limit
time_limit_per_trial = args.time_limit_per_trial
seed = 4774
print(ip, port, n_workers)
print(R, eta)


def get_cs():
    cs = ConfigurationSpace()
    rate = UniformFloatHyperparameter("rate", 0.0, 5.0)
    num = UniformIntegerHyperparameter("num", 0, 99, default_value=50)
    cs.add_hyperparameters([rate, num])
    return cs


def evaluate_parallel(method_id, n_workers, seed, ip, port):
    print(method_id, n_workers, seed)
    if port == 0:
        port = 13579 + np.random.randint(1000)
    print('ip=', ip, 'port=', port)

    def mf_objective_func(config, n_resource, extra_conf):
        obj_seed = int(time.time() * 10000) % 100000
        rng = np.random.RandomState(obj_seed)
        perf = -rng.random() * 100
        result = dict(
            objective_value=perf,
            early_stop=False,  # for deep learning
            ref_id=None,
        )
        return result

    cs = get_cs()

    def master_run(return_list):
        asha = async_mqSuccessiveHalving(None, cs, R, eta=eta,
                                         random_state=seed,
                                         method_id=method_id, restart_needed=True,
                                         time_limit_per_trial=time_limit_per_trial,
                                         runtime_limit=runtime_limit,
                                         ip='', port=port,
                                         )
        asha.run()
        print('===== incumbent =====')
        for config, perf in zip(asha.incumbent_configs, asha.incumbent_perfs):
            print(config, perf)
        return_list.extend(asha.recorder)  # send to return list

    def worker_run(i):
        worker = async_mqmfWorker(mf_objective_func, ip, port)
        worker.run()
        print("Worker %d exit." % (i,))

    manager = Manager()
    recorder = manager.list()   # shared list
    master = Process(target=master_run, args=(recorder,))
    master.start()

    time.sleep(3)  # wait for master init
    worker_pool = []
    for i in range(n_workers):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:   # optional if repeat=1
        w.join()

    return list(recorder)   # covert to list


method_id = 'test_asha-n%d-%04d' % (n_workers, seed)
evaluate_parallel(method_id, n_workers, seed, ip, port)

