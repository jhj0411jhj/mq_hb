"""
example cmdline:

python test/default_test/xgb_default.py --dataset spambase --n_jobs 15 --seed 47

"""

import os
import time
import argparse
import numpy as np
import pickle as pkl
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')    # for dependency
from test.utils import load_data, timeit

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_jobs', type=int, default=15)
parser.add_argument('--seed', type=int, default=47)

args = parser.parse_args()
dataset = args.dataset
n_jobs = args.n_jobs
seed = args.seed


def objective_func(x_train, x_val, x_test, y_train, y_val, y_test):
    eval_s = [(x_train, y_train), (x_val, y_val)]   # to get evals_result

    t0 = time.time()
    model = XGBClassifier(
        use_label_encoder=False,
        random_state=np.random.RandomState(seed),
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train, eval_set=eval_s)
    train_time = time.time() - t0

    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize

    # evaluate on test data
    y_pred_test = model.predict(x_test)
    test_perf = -balanced_accuracy_score(y_test, y_pred_test)  # minimize

    try:
        evals_result = model.evals_result()
    except Exception:
        evals_result = None

    return perf, test_perf, evals_result, train_time


x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)

with timeit('%s-%d' % (dataset, seed)):
    print('=== xgb default param ===')
    perf, test_perf, evals_result, train_time = objective_func(x_train, x_val, x_test, y_train, y_val, y_test)
    print(evals_result)
    print('=== perf(val, test):', perf, test_perf)
    print('=== train time(s):', train_time)
    # print(list(evals_result['validation_0'].values())[0])

    config = None
    save_item = (config, perf, test_perf, evals_result, train_time)

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    dir_path = 'data/default_test/xgb-%s/' % (dataset, )
    file_name = 'xgb_default-%s-%04d-%s.pkl' % (dataset, seed, timestamp)
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        pass
    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pkl.dump(save_item, f)
    print('save to:', dir_path, file_name)
