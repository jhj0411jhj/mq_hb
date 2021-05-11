"""
example cmdline:

python test/test_xgb_model.py --dataset spambase --n_jobs 4 --rep 1

"""

import argparse
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.append('.')
sys.path.insert(0, '../open-box')    # for dependency
from mq_hb.xgb_model import XGBoost
from mq_hb.utils import sample_configurations
from utils import load_data, timeit

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--rep', type=int, default=10)

args = parser.parse_args()
dataset = args.dataset
n_jobs = args.n_jobs
rep = args.rep


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = XGBoost(**conf_dict, n_jobs=n_jobs, seed=seed)
    model.fit(x_train, y_train)

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
cs.seed(seed)

x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)

configs = sample_configurations(cs, rep)
with timeit('%s-all' % (dataset,)):
    for i in range(rep):
        with timeit('%s-%d' % (dataset, i)):
            conf = configs[i]
            result = objective_func(conf, x_train, x_val, y_train, y_val)
            print(result['objective_value'])
            print(conf)
