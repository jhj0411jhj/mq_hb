import contextlib
import os
import time
import argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.append('.')
sys.path.insert(0, '../lite-bo')    # for dependency

from mq_hb.xgb_model import XGBoost
from mq_hb.mq_hb import sample_configurations

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--rep', type=int, default=10)

args = parser.parse_args()
dataset = args.dataset
n_jobs = args.n_jobs
rep = args.rep


# timer tool
@contextlib.contextmanager
def timeit(name=''):
    print("[%s]Start." % name)
    start = time.time()
    yield
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("[%s]Total time = %d hours, %d minutes, %d seconds." % (name, h, m, s))


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


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = XGBoost(**conf_dict, n_jobs=n_jobs)
    model.fit(x_train, y_train)

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
