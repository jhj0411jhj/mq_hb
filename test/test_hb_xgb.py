import os
import argparse
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.append('.')
sys.path.insert(0, '../lite-bo')    # for dependency

from mq_hb.mq_hb import mqHyperband
from mq_hb.mq_mf_worker import mqmfWorker
from mq_hb.xgb_model import XGBoost


parser = argparse.ArgumentParser()
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int)
parser.add_argument('--dataset', type=str)

parser.add_argument('--num_iter', type=int, default=1)
parser.add_argument('--runtime_limit', type=int)

args = parser.parse_args()
role = args.role
ip = args.ip
port = args.port
n_jobs = args.n_jobs
R = args.R
eta = args.eta
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
dataset = args.dataset
num_iter = args.num_iter
runtime_limit = args.runtime_limit
print(role, ip, port, n_jobs, n_workers, dataset)
print(R, eta)
print(num_iter, runtime_limit)
for para in (role, ip, port, n_jobs, R, eta, n_workers):
    assert para is not None


def load_data(dataset):
    data_dir = 'datasets'
    name_x_train = dataset + '-x_train.npy'
    name_x_val = dataset + '-x_val.npy'
    name_x_test = dataset + '-x_test.npy'
    name_y_train = dataset + '-y_train.npy'
    name_y_val = dataset + '-y_val.npy'
    name_y_test = dataset + '-y_test.npy'
    x_train = np.load(os.path.join(data_dir, name_x_train))
    x_val = np.load(os.path.join(data_dir, name_x_val))
    x_test = np.load(os.path.join(data_dir, name_x_test))
    y_train = np.load(os.path.join(data_dir, name_y_train))
    y_val = np.load(os.path.join(data_dir, name_y_val))
    y_test = np.load(os.path.join(data_dir, name_y_test))
    print(dataset, 'loaded. n_instances =', x_train.shape[0], x_val.shape[0], x_test.shape[0])
    return x_train, x_val, x_test, y_train, y_val, y_test


def mf_objective_func(config: dict, n_resource, total_resource, x_train, x_val, y_train, y_val, seed):
    uid = config.pop('uid', 1)
    reference = config.pop('reference', None)
    need_lc = config.pop('need_lc', None)
    method_name = config.pop('method_name', None)
    print('objective extra info in config:', uid, reference, need_lc, method_name)

    # sample train data. the test data after split is sampled train data
    if n_resource < total_resource:
        ratio = n_resource / total_resource
        print('sample data: ratio =', ratio, n_resource, total_resource)
        _x, sample_x, _y, sample_y = train_test_split(x_train, y_train, test_size=ratio,
                                                      stratify=y_train, random_state=seed)
    else:
        print('sample data: use full dataset', n_resource, total_resource)
        sample_x, sample_y = x_train, y_train

    model = XGBoost(**config, n_jobs=n_jobs)
    model.fit(sample_x, sample_y)

    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = 1 - balanced_accuracy_score(y_val, y_pred)

    result = dict(
        objective_value=perf,
        early_stop=False,   # for deep learning
        ref_id=None,
    )
    return result


cs = XGBoost.get_cs()
seed = 123

if role == 'master':
    method_id = 'hyperband-n%d-%s-%04d' % (n_workers, dataset, seed)
    hyperband = mqHyperband(None, cs, R, eta=eta,
                            num_iter=num_iter, random_state=seed,
                            method_id=method_id, restart_needed=True,
                            time_limit_per_trial=600, ip='', port=port)
    hyperband.runtime_limit = runtime_limit     # set total runtime limit
    hyperband.run()
else:
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)
    mf_objective_func = partial(mf_objective_func, total_resource=R, seed=seed,
                                x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)
    worker = mqmfWorker(mf_objective_func, ip, port)
    worker.run()
