import os
import time
import contextlib
import numpy as np


def load_data(dataset, data_dir='datasets'):
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
