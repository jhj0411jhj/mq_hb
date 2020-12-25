"""
example cmdline:

python test_hb_xgb.py --role master --ip '' --n_worker 1 --dataset spambase --R 27
python test_hb_xgb.py --role worker --ip 127.0.0.1 --n_worker 1 --dataset spambase --R 27

"""

import argparse
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.append('.')
sys.path.insert(0, '../lite-bo')    # for dependency
from mq_hb.mq_hb import mqHyperband
from mq_hb.mq_mf_worker import mqmfWorker
from mq_hb.xgb_model import XGBoost
from utils import load_data

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

    model = XGBoost(**params, n_jobs=n_jobs)
    model.fit(sample_x, sample_y)

    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize

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
