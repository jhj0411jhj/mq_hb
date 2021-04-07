import os
import time
import contextlib
import numpy as np
import pickle as pkl

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


def setup_exp(_dataset, n_jobs, runtime_limit, time_limit_per_trial):
    if _dataset == 'kuaishou1':
        n_jobs = 8
        runtime_limit = 24 * 3600           # 24h
        time_limit_per_trial = 1 * 3600     # 1h
    elif _dataset == 'kuaishou2':
        n_jobs = 8
        runtime_limit = 48 * 3600           # 48h
        time_limit_per_trial = 4 * 3600     # 4h
    elif _dataset == 'mnist_784':
        n_jobs = 4
        runtime_limit = 12 * 3600           # 12h
        time_limit_per_trial = 3 * 3600     # 3h
    elif _dataset == 'HIGGS':
        n_jobs = 4
        runtime_limit = 12 * 3600           # 12h
        time_limit_per_trial = 999999
    elif _dataset == 'covertype':
        n_jobs = 4
        runtime_limit = 5 * 3600            # 5h
        time_limit_per_trial = 1200         # 20min
    elif _dataset == 'codrna':
        n_jobs = 4
        runtime_limit = 3000                # 50min
        time_limit_per_trial = 1200         # 20min
    else:
        print('[setup exp] dataset setup not found. use input settings.')
    print('[setup exp] dataset=%s, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
          % (_dataset, n_jobs, runtime_limit, time_limit_per_trial))
    for para in (n_jobs, runtime_limit, time_limit_per_trial):
        assert para is not None and para > 0
    return n_jobs, runtime_limit, time_limit_per_trial


def load_data(dataset, data_dir='datasets', **kwargs):
    if dataset.startswith('kuaishou'):
        from ks.ks_utils import load_ks_data
        return load_ks_data(dataset, data_dir=data_dir, **kwargs)

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
    print(dataset, 'loaded. x shape =', x_train.shape, x_val.shape, x_test.shape)
    return x_train, x_val, x_test, y_train, y_val, y_test


def check_datasets(datasets):
    for _dataset in datasets:
        if _dataset.startswith('kuaishou'):     # too large. don't check
            continue
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


# ===== for plot =====

def descending(x):
    y = [x[0]]
    for i in range(1, len(x)):
        y.append(min(y[-1], x[i]))
    return y


def create_point(x, stats, default=0.0):
    """
    get the closest perf of time point x where timestamp < x
    :param x:
        the time point
    :param stats:
        list of func. func is tuple of timestamp list and perf list
    :param default:
        init value of perf
    :return:
        list of perf of funcs at time point x
    """
    perf_list = []
    for func in stats:
        timestamp, perf = func
        last_p = default
        for t, p in zip(timestamp, perf):
            if t > x:
                break
            last_p = p
        perf_list.append(last_p)
    return perf_list


def create_plot_points(stats, start_time, end_time, point_num=500):
    """

    :param stats:
        list of func. func is tuple of timestamp list and perf list
    :param start_time:
    :param end_time:
    :param point_num:
    :return:
    """
    x = np.linspace(start_time, end_time, num=point_num)
    _mean, _std = list(), list()
    for i, stage in enumerate(x):
        perf_list = create_point(stage, stats)
        _mean.append(np.mean(perf_list))
        _std.append(np.std(perf_list))
    # Used to plot errorbar.
    return x, np.array(_mean), np.array(_std)


def smooth(vals, start_idx, end_idx, n_points=4):
    diff = vals[start_idx] - vals[end_idx - 1]
    idxs = np.random.choice(list(range(start_idx, end_idx)), n_points)
    new_vals = vals.copy()
    val_sum = 0.
    new_vals[start_idx:end_idx] = vals[start_idx]
    for idx in sorted(idxs):
        _val = np.random.uniform(0, diff * 0.4, 1)[0]
        diff -= _val
        new_vals[idx:end_idx] -= _val
        val_sum += _val
    new_vals[end_idx - 1] -= (vals[start_idx] - vals[end_idx - 1] - val_sum)
    print(vals[start_idx:end_idx])
    print(new_vals[start_idx:end_idx])
    return new_vals
