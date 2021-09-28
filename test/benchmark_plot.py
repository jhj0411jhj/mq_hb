"""
run benchmark_process_record.py first to get new_record file

example cmdline:

python test/benchmark_plot.py --dataset covtype --R 27

"""
import argparse
import os
import time
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
parser.add_argument('--base_time', type=int)    # plot method with different n_workers and runtime_limit
parser.add_argument('--model', type=str, default='xgb')
parser.add_argument('--default_value', type=float, default=0.0)

args = parser.parse_args()
dataset = args.dataset
mths = args.mths.split(',')
R = args.R
model = args.model
base_time = args.base_time
default_value = args.default_value


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


def get_mth_legend(name, show_mode=False):
    if not show_mode:
        return mth
    name = name.lower()

    if name.startswith('asha') and '_stop' in name:
        label = 'ASHA-Stop'
    elif name.startswith('asha'):
        label = 'ASHA'
    elif name.startswith('ahb') and '_stop' in name:
        label = 'A-HB-Stop'
    elif name.startswith('ahb'):
        label = 'A-HB'
    elif name.startswith('ams'):
        label = 'A-MS'
    elif name.startswith('amfes') and '_stop' in name:
        label = 'A-MFES-Stop'
    elif name.startswith('amfes_ms'):
        label = 'A-MFES-MS'
    elif name.startswith('amfes'):
        label = 'A-MFES'
    elif name.startswith('amfprf_ms'):
        label = 'A-MFPRF-MS'
    elif name.startswith('mfes'):
        label = 'MFES'
    else:
        label = name
    return label


def plot_setup(_dataset):
    y_range = None
    if _dataset == 'covtype':
        y_range = [-0.940, -0.880]
    elif _dataset == 'pokerhand':
        y_range = [-1.001, -0.951]
    elif _dataset.startswith('HIGGS'):
        y_range = [-0.7555, -0.7485]
    elif _dataset.startswith('hepmass'):
        y_range = [-0.8755, -0.8725]
    elif _dataset.startswith('censusincome'):
        y_range = [-0.747, -0.737]
    elif _dataset == 'codrna':
        y_range = [-0.9793, -0.9753]
    elif _dataset == 'kuaishou1':
        y_range = [-0.7717, -0.7709]
    elif _dataset == 'kuaishou2':
        y_range = [-0.636, -0.611]
    elif _dataset == 'cifar10-valid':
        y_range = [-91.60, -90.80]
    elif _dataset == 'cifar100':
        y_range = [-73.7, -70.7]
    elif _dataset == 'ImageNet16-120':
        y_range = [-47.0, -45.0]
    elif _dataset == 'mfeat-fourier(1)':
        y_range = [-0.835, -0.810]
    elif _dataset == 'cifar10':
        y_range = [-0.935, -0.925]
    elif _dataset == 'penn':
        y_range = [65, 77]

    if y_range is not None:
        plt.ylim(*y_range)
    plt.xlim(0, base_time or runtime_limit)


print('start', dataset)
# setup
_, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
if args.runtime_limit is not None:
    runtime_limit = args.runtime_limit
plot_setup(dataset)
color_dict, marker_dict = fetch_color_marker(mths)
point_num = 300
lw = 2
markersize = 6
markevery = int(point_num / 10)
std_scale = 0.5
std_alpha = 0.15

plot_list = []
legend_list = []
result = dict()
for mth in mths:
    if base_time is not None:
        n_workers = int(mth.split('-')[-1][1:])     # 'ALGO-nX'
        runtime_limit = int(base_time / n_workers)
        print('Plotting %s: n_workers=%d, base_time=%d, runtime_limit=%d.' % (mth, n_workers, base_time, runtime_limit))
    stats = []
    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('new_record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                raw_recorder = pkl.load(f)
            recorder = []
            for record in raw_recorder:
                # if record.get('n_iteration') is not None and record['n_iteration'] < R:
                #     print('error abandon record by n_iteration:', R, mth, record)
                #     continue
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
    x, m, s = create_plot_points(stats, 0, runtime_limit, point_num=point_num, default=default_value)
    result[mth] = (x, m, s)
    # plot
    plt.plot(x, m, lw=lw, label=get_mth_legend(mth, show_mode=False),
             #color=color_dict[mth], marker=marker_dict[mth],
             markersize=markersize, markevery=markevery)
    #plt.fill_between(x, m - s * std_scale, m + s * std_scale, alpha=alpha, facecolor=color_dict[mth])

# calculate speedup
speedup_algo = 2
print('===== mth - baseline - speedup ===== speedup_algo =', speedup_algo)
for mth in mths:
    if not (mth.startswith('amfes') or mth.startswith('mfes')):
        continue
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

# print last val perf
print('===== mth - last val perf =====')
for mth in mths:
    x, m, s = result[mth]
    m = m[-1]
    s = s[-1]
    perfs = None
    if dataset == 'kuaishou1':
        print(dataset, mth, perfs, u'%.5f\u00B1%.5f' % (m, s))
    elif dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
        print(dataset, mth, perfs, u'%.2f\u00B1%.2f' % (m, s))
    else:
        print(dataset, mth, perfs, u'%.4f\u00B1%.4f' % (m, s))

# plt.axhline(-0.849296, linestyle="--", color="b", lw=1, label="Default")
# plt.ylim(-0.863, -0.844)
# plt.xlim(0, runtime_limit+1000)

# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1, 64)
# plt.xlim(10, base_time*1.1)

# show plot
plt.legend(loc='upper right')
plt.title("%s on %s" % (model, dataset), fontsize=16)
plt.xlabel("Wall Clock Time (sec)", fontsize=16)
plt.ylabel("Validation Error", fontsize=16)
plt.tight_layout()
plt.grid()
plt.show()
