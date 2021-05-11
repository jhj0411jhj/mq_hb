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

#default_mths = 'random-n1,random-n3,smac,hyperband-n1,hyperband-n3,bohb-n1,bohb-n3,mfes-n1,mfes-n3'
default_mths = 'hyperband-n8,bohb-n8,mfes-n8,amfesv0-n8,amfesv3-n8,amfesv6-n8'

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


def fetch_color_marker_old(m_list):
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
        elif name.startswith('bohb-n'):
            fill_values(name, 4)
        elif name.startswith('mfes-n1'):
            fill_values(name, 8)
        elif name.startswith('mfes-n'):
            fill_values(name, 7)
        elif name.startswith('amfes-n1'):
            fill_values(name, 2)
        elif name.startswith('amfes-n'):
            fill_values(name, 1)
        elif name.startswith('mfesv2-n1'):
            fill_values(name, 3)
        elif name.startswith('mfesv2-n3'):
            fill_values(name, 5)
        elif name.startswith('mfesv3-n1'):
            fill_values(name, 2)
        elif name.startswith('mfesv3-n3'):
            fill_values(name, 1)
        else:
            print('color not defined:', name)
            fill_values(name, 1)
    return color_dict, marker_dict


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'black', 'yellow']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', 'd']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('random'):
            fill_values(name, 0)
        elif name.startswith('smac'):
            fill_values(name, 6)
        # elif name.startswith('sh'):
        #     fill_values(name, 2)
        elif name.startswith('hyperband'):
            fill_values(name, 2)
        elif name.startswith('bohb'):
            fill_values(name, 1)
        elif name.startswith('mfes'):
            fill_values(name, 5)
        # elif name.startswith('asha'):
        #     fill_values(name, 7)
        elif name.startswith('ahb'):
            fill_values(name, 7)
        # elif name.startswith('abohb'):
        #     fill_values(name, 8)
        elif name.startswith('amfes'):
            fill_values(name, 4)
        else:
            print('color not defined:', name)
            fill_values(name, 1)
    return color_dict, marker_dict


def get_mth_legend(mth, show_mode=False):
    mth_lower = mth.lower()
    legend_dict = {
        'random-n1': 'Random-n1',
        'smac': 'SMAC',
        'hyperband-n1': 'Hyperband-n1',
        'bohb-n1': 'BOHB-n1',
        'mfes-n1': 'MFES-n1',

        'hyperband-n8': 'Hyperband-n8',
        'bohb-n8': 'BOHB-n8',
        'mfes-n8': 'MFES-n8',

        'random-n3': 'Random-n3',
        'hyperband-n3': 'Hyperband-n3',
        'bohb-n3': 'BOHB-n3',
        'mfes-n3': 'MFES-n3',
    }
    if show_mode:
        if mth.startswith('amfes') and mth.endswith('-n8'):
            mth = 'AMFES-n8'
        if mth.startswith('mfes') and mth.endswith('-n8'):
            mth = 'MFES-n8'
        legend_dict['ahb-n1'] = 'ASHA-n1'
        legend_dict['ahb-n8'] = 'ASHA-n8'
    return legend_dict.get(mth_lower, mth)


def plot_setup(_dataset):
    if _dataset == 'covtype':
        plt.ylim(-0.937, -0.877)
        plt.xlim(0, runtime_limit+10)
    elif _dataset == 'codrna':
        plt.ylim(-0.9793, -0.9753)
        plt.xlim(0, runtime_limit+10)
    elif _dataset == 'kuaishou1':
        plt.ylim(-0.7717, -0.7709)
        plt.xlim(0, runtime_limit+1000)
    elif _dataset == 'kuaishou2':
        plt.ylim(-0.636, -0.611)
        plt.xlim(0, runtime_limit+1000)
    elif _dataset == 'pokerhand':
        plt.ylim(-1.001, -0.951)
        plt.xlim(0, runtime_limit+10)
    elif _dataset.startswith('HIGGS'):
        plt.ylim(-0.756, -0.746)
        plt.xlim(0, runtime_limit+10)
    elif _dataset.startswith('hepmass'):
        plt.ylim(-0.8755, -0.8725)
        plt.xlim(0, runtime_limit+10)
    elif _dataset.startswith('censusincome'):
        plt.ylim(-0.747, -0.737)
        plt.xlim(0, runtime_limit)


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
                if record.get('n_iteration') is not None and record['n_iteration'] < R:
                    print('error abandon record by n_iteration:', R, mth, record)
                    continue
                if record['global_time'] > runtime_limit:
                    print('abandon record by runtime_limit:', runtime_limit, mth, record)
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
    plt.plot(x, m, lw=lw, label=get_mth_legend(mth, show_mode=True),
             color=color_dict[mth], marker=marker_dict[mth],
             markersize=markersize, markevery=markevery)
    #plt.fill_between(x, m - s * std_scale, m + s * std_scale, alpha=alpha, facecolor=color_dict[mth])

# calculate speedup
speedup_algo = 2
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
        print("%s %s %.2f" % (mth, baseline, speedup))

# plt.axhline(-0.849296, linestyle="--", color="b", lw=1, label="Default")
# plt.ylim(-0.863, -0.844)
# plt.xlim(0, runtime_limit+1000)

# show plot
plt.legend(loc='upper right')
plt.title(dataset, fontsize=16)
plt.xlabel("Time elapsed (sec)", fontsize=16)
plt.ylabel("Negative validation score", fontsize=16)
plt.tight_layout()
plt.show()
