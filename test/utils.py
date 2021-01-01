import os
import time
import contextlib
import numpy as np
import pickle as pkl

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


def setup_exp(_dataset, n_jobs, runtime_limit, time_limit_per_trial):
    if _dataset == 'mnist_784':
        n_jobs = 4
        runtime_limit = 12 * 3600           # 12h
        time_limit_per_trial = 3 * 3600     # 3h
    elif _dataset == 'higgs':
        n_jobs = 4
        runtime_limit = 2 * 3600            # 2h
        time_limit_per_trial = 600          # 10min
    elif _dataset == 'covertype':
        n_jobs = 4
        runtime_limit = 5 * 3600            # 5h
        time_limit_per_trial = 1200         # 20min
    elif _dataset == 'covtype':
        n_jobs = 4
        runtime_limit = 12 * 3600           # 12h
        time_limit_per_trial = 2 * 3600     # 2h
    elif _dataset == 'codrna':
        n_jobs = 4
        runtime_limit = 1 * 3600            # 1h
        time_limit_per_trial = 2 * 3600     # 2h
    else:
        print('[setup exp] dataset setup not found. use input settings.')
    print('[setup exp] dataset=%s, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
          % (_dataset, n_jobs, runtime_limit, time_limit_per_trial))
    for para in (n_jobs, runtime_limit, time_limit_per_trial):
        assert para is not None and para > 0
    return n_jobs, runtime_limit, time_limit_per_trial


def load_data(dataset, data_dir='datasets'):
    if dataset == 'codrna':
        name = dataset + '.pkl'
        with open(os.path.join(data_dir, name), 'rb') as f:
            obj = pkl.load(f)
            x_train, x_val, x_test, y_train, y_val, y_test = obj
    else:
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


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset)
        except Exception as e:
            print('Dataset - %s load error: %s' % (_dataset, str(e)))
            raise


# timer tool
@contextlib.contextmanager
def timeit(name=''):
    print("[%s]Start." % name, flush=True)
    start = time.time()
    yield
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("[%s]Total time = %d hours, %d minutes, %d seconds." % (name, h, m, s), flush=True)
