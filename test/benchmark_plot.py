"""
run benchmark_process_record.py first to get new_record file

example cmdline:

python test/benchmark_plot.py --datasets covtype,codrna

"""
import argparse
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import setup_exp

# default_datasets = 'mnist_784,higgs,covertype'
default_datasets = 'covtype,codrna'
default_mths = 'random-n1,random-n3,smac,hyperband-n1,hyperband-n3'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--R', type=int, default=27)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
mths = args.mths.split(',')
R = args.R
model = 'xgb'


def descending(x):
    def _descending(x):
        y = [x[0]]
        for i in range(1, len(x)):
            y.append(min(y[-1], x[i]))
        return y

    if isinstance(x[0], list):
        y = []
        for xi in x:
            y.append(_descending(xi))
        return y
    else:
        return _descending(x)


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


def plot_setup(_dataset):
    if _dataset == 'covtype':
        plt.ylim(-0.925, -0.84)
    elif _dataset == 'codrna':
        plt.ylim(-0.979, -0.974)


for dataset in test_datasets:
    print('start', dataset)
    # setup
    _, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
    plot_setup(dataset)

    plot_list = []
    legend_list = []
    result = dict()
    for mth in mths:
        stats = []
        dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
        for file in os.listdir(dir_path):
            if file.startswith('new_record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
                # if mth == 'smac' and (file.find('6295') >= 0 or file.find('4531') >= 0):
                #     print('abandon', mth, file)
                #     continue  # todo
                with open(os.path.join(dir_path, file), 'rb') as f:
                    raw_recorder = pkl.load(f)
                recorder = []
                for record in raw_recorder:
                    if record.get('n_iteration') is not None and record['n_iteration'] < R:
                        if not mth.startswith('hyperband'):
                            print('error abandon record by n_iteration:', R, mth, record)
                        continue
                    if record['global_time'] > runtime_limit:
                        # print('abandon record by runtime_limit:', runtime_limit, mth, record)
                        continue
                    recorder.append(record)
                recorder.sort(key=lambda rec: rec['global_time'])
                # print([(rec['global_time'], rec['return_info']['loss']) for rec in recorder])
                print('new recorder len:', mth, len(recorder), len(raw_recorder))
                timestamp = [rec['global_time'] for rec in recorder]
                perf = descending([rec['return_info']['loss'] for rec in recorder])
                stats.append((timestamp, perf))
                # if mth == 'smac':
                #     plt.plot(timestamp, perf, label=file)
        x, m, s = create_plot_points(stats, 0, runtime_limit, 10000)
        result[mth] = (x, m, s)
        # plot
        plt.plot(x, m, label=mth)
        #std_scale = 1
        #plt.fill_between(x, m - s * std_scale, m + s * std_scale, alpha=0.2)

    # calculate speedup
    print('===== mth - baseline - speedup')
    for mth in mths:
        for baseline in mths:
            baseline_perf = result[baseline][1][-1]
            baseline_time = result[baseline][0][-1]
            x, m, s = result[mth]
            x, m, s = x.tolist(), m.tolist(), s.tolist()
            mth_time = baseline_time    # if speedup < 1: speedup = 1
            for xi, mi, si in zip(x, m, s):
                if mi <= baseline_perf:
                    mth_time = xi
                    break
            speedup = baseline_time / mth_time
            print(mth, baseline, speedup)

    # show plot
    plt.legend(loc='upper right')
    plt.title(dataset, fontsize=16)
    plt.xlabel("Time elapsed (sec)", fontsize=16)
    plt.ylabel("Negative validation score", fontsize=16)
    plt.tight_layout()
    plt.show()
