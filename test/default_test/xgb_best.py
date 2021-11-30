"""
example cmdline:

python test/default_test/xgb_best.py --dataset spambase --n_jobs 15 --mth amfesv20-n8 --rep 5 --start_id 0

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
from mq_hb.xgb_model import XGBoost
from test.utils import load_data, timeit, seeds, setup_exp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mth', type=str, default='amfesv20-n8')
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--runtime_limit', type=int)    # if you don't want to use default setup
parser.add_argument('--n_jobs', type=int, default=15)
parser.add_argument('--seed', type=int, default=47)

args = parser.parse_args()
dataset = args.dataset
mth = args.mth
rep = args.rep
start_id = args.start_id
# runtime_limit = args.runtime_limit
# n_jobs = args.n_jobs
xgb_seed = args.seed


def objective_func(config, x_train, x_val, x_test, y_train, y_val, y_test):
    eval_s = [(x_train, y_train), (x_val, y_val)]   # to get evals_result

    t0 = time.time()
    conf_dict = config.get_dictionary()
    model = XGBoost(**conf_dict, n_jobs=n_jobs, seed=xgb_seed)
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


# setup
n_jobs, runtime_limit, _ = setup_exp(dataset, 4, 1, 1)
if args.runtime_limit is not None:
    runtime_limit = args.runtime_limit
if args.n_jobs is not None:
    n_jobs = args.n_jobs
x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)
print('===== start test %s %s: rep=%d, n_jobs=%d' % (mth, dataset, rep, n_jobs))
for i in range(start_id, start_id + rep):
    seed = seeds[i]

    dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('incumbent_new_record_%s-%s-%d-' % (mth, dataset, seed)) \
                and file.endswith('.pkl'):
            # load config
            with open(os.path.join(dir_path, file), 'rb') as f:
                record = pkl.load(f)
            print(dataset, mth, seed, 'loaded!', record, flush=True)

            # run test
            config = record['configuration']
            print('=== xgb best param ===')
            perf, test_perf, evals_result, train_time = objective_func(config, x_train, x_val, x_test, y_train, y_val,
                                                                       y_test)
            print(evals_result)
            print('=== perf(val, test):', perf, test_perf)
            print('=== train time(s):', train_time)
            # print(list(evals_result['validation_0'].values())[0])

            save_item = (config, perf, test_perf, evals_result, train_time)

            # save perf
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            method_id = mth + '-%s-%d-%s' % (dataset, seed, timestamp)
            save_dir_path = 'data/default_test/xgb-%s/' % (dataset, )
            save_file_name = 'xgb_best-%s.pkl' % (method_id,)
            try:
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
            except FileExistsError:
                pass
            with open(os.path.join(save_dir_path, save_file_name), 'wb') as f:
                pkl.dump(save_item, f)
            print('save to:', save_dir_path, save_file_name)

