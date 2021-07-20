"""
example cmdline:

python test/test_asha_stopping.py --role master --n_workers 1 --runtime_limit 60
python test/test_asha_stopping.py --role worker --ip 127.0.0.1 --port 13579 --self_ip 127.0.0.1 --self_port 13531

"""

import argparse
import numpy as np
import time

import sys
sys.path.append('.')
sys.path.insert(0, '../open-box')    # for dependency
from mq_hb.async_mq_sh_stopping import async_mqSuccessiveHalving_stopping
from mq_hb.async_mq_mf_worker_stopping import async_mqmfWorker_stopping
from mq_hb.message_queue.reporter import Reporter

from openbox.utils.config_space import ConfigurationSpace, Configuration, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--self_ip', type=str, default='127.0.0.1')
parser.add_argument('--self_port', type=int, default=13531)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--runtime_limit', type=int, default=60)
parser.add_argument('--role', type=str, choices=['master', 'worker'])

args = parser.parse_args()
ip = args.ip
port = args.port
self_ip = args.self_ip
self_port = args.self_port
R = args.R
eta = args.eta
n_workers = args.n_workers
runtime_limit = args.runtime_limit
role = args.role
print(ip, port, n_workers)
print(R, eta)
print(self_ip, self_port)


def get_cs():
    cs = ConfigurationSpace()
    rate = UniformFloatHyperparameter("rate", 0.0, 5.0)
    num = UniformIntegerHyperparameter("num", 0, 99, default_value=50)
    cs.add_hyperparameters([rate, num])
    return cs

cs = get_cs()


def mf_objective_func(config, n_resource, extra_conf, reporter: Reporter):
    obj_seed = int(time.time() * 10000) % 100000
    rng = np.random.RandomState(obj_seed)

    print('worker config:', config)
    next_n_iteration = n_resource

    while True:
        start_time = time.time()

        time.sleep(1)
        perf = -rng.random() * 100
        print('worker iter:', next_n_iteration, 'perf:', perf)

        next_n_iteration = reporter(
            objective_value=perf,
            n_iteration=next_n_iteration,
            time_taken=time.time() - start_time,
            test_perf=None,
        )


if role == 'master':
    algo = async_mqSuccessiveHalving_stopping(
        None, cs, R, eta=eta,
        random_state=47,
        method_id='test_asha_stopping', restart_needed=True,
        time_limit_per_trial=999999,
        runtime_limit=runtime_limit,
        ip='', port=port,
    )
    algo.run()
    print(algo.get_bracket_status(algo.bracket))
else:
    worker = async_mqmfWorker_stopping(
        mf_objective_func,
        ip=ip, port=port, authkey=b'abc',
        self_ip=self_ip, self_port=self_port, self_authkey=b'abc',
    )
    worker.run()
