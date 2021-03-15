"""
run benchmark_process_record.py first to get new_record file

example cmdline:

python test/benchmark_plot.py --dataset covtype --R 27

"""
import argparse
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import setup_exp, descending, create_plot_points

default_mths = 'random-n1,random-n3,smac,hyperband-n1,hyperband-n3,bohb-n1,bohb-n3'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--runtime_limit', type=int)    # if you don't want to use default setup

args = parser.parse_args()
dataset = args.dataset
mths = args.mths.split(',')
R = args.R
model = 'xgb'


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'black', 'yellow']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', 'd']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('random-n1'):
            fill_values(name, 1)
        elif name.startswith('random-n3'):
            fill_values(name, 6)
        elif name.startswith('smac'):
            fill_values(name, 3)
        elif name.startswith('hyperband-n1'):
            fill_values(name, 5)
        elif name.startswith('hyperband-n3'):
            fill_values(name, 2)
        elif name.startswith('bohb-n1'):
            fill_values(name, 0)
        elif name.startswith('bohb-n3'):
            fill_values(name, 4)
        elif name.startswith('other'):
            fill_values(name, 7)
        else:
            print('color not defined:', name)
            fill_values(name, 8)
    return color_dict, marker_dict


def get_mth_legend(mth):
    mth = mth.lower()
    if mth == 'random-n1':
        return 'Random-n1'
    elif mth == 'random-n3':
        return 'Random-n3'
    elif mth == 'smac':
        return 'SMAC'
    elif mth == 'hyperband-n1':
        return 'Hyperband-n1'
    elif mth == 'hyperband-n3':
        return 'Hyperband-n3'
    elif mth == 'bohb-n1':
        return 'BOHB-n1'
    elif mth == 'bohb-n3':
        return 'BOHB-n3'
    else:
        return mth


def plot_setup(_dataset):
    if _dataset == 'covtype':
        plt.ylim(-0.93, -0.87)
        plt.xlim(0, runtime_limit+200)
    elif _dataset == 'codrna':
        plt.ylim(-0.9793, -0.9753)
        plt.xlim(0, runtime_limit+10)
    elif _dataset == 'kuaishou1':
        plt.ylim(-0.7717, -0.7709)
        plt.xlim(0, runtime_limit+1000)
    elif _dataset == 'kuaishou2':
        plt.ylim(-0.636, -0.611)
        plt.xlim(0, runtime_limit+1000)


print('start', dataset)
# setup
_, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
if args.runtime_limit is not None:
    runtime_limit = args.runtime_limit
plot_setup(dataset)
color_dict, marker_dict = fetch_color_marker(mths)
lw = 2
markersize = 6
markevery = int(10000 / 10)
std_scale = 0.3
alpha = 0.2

plot_list = []
legend_list = []
result = dict()
for mth in mths:
    stats = []
    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('new_record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                raw_recorder = pkl.load(f)
            recorder = []
            for record in raw_recorder:
                # assert len(record['configuration'].get_dictionary()) == 9
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
            # for debugging
            # if mth == 'smac':
            #     plt.plot(timestamp, perf, label=file)
    x, m, s = create_plot_points(stats, 0, runtime_limit, 10000)
    result[mth] = (x, m, s)
    # plot
    plt.plot(x, m, lw=lw, label=get_mth_legend(mth), color=color_dict[mth],
             marker=marker_dict[mth], markersize=markersize, markevery=markevery)
    #plt.fill_between(x, m - s * std_scale, m + s * std_scale, alpha=alpha, facecolor=color_dict[mth])

# calculate speedup
speedup_algo = 1
print('===== mth - baseline - speedup ===== speedup_algo =', speedup_algo)
for mth in mths:
    for baseline in mths:
        baseline_perf = result[baseline][1][-1]
        if speedup_algo == 1:   # algo 1
            baseline_time = None
            x, m, s = result[baseline]
            x, m, s = x.tolist(), m.tolist(), s.tolist()
            for xi, mi, si in zip(x, m, s):
                if mi <= baseline_perf:
                    baseline_time = xi
                    break
            assert baseline_time is not None
        else:   # algo 2: baseline_time is last record time
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
