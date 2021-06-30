"""
plot curve
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
import pylab
from math import ceil
from scipy.interpolate import make_interp_spline

# sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)
# plt.rc('font', size=18.0, family='Times New Roman')
plt.rc('font', size=18.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['figure.figsize'] = (8.0, 3.0)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# pylab.rcParams['figure.figsize'] = (8.0, 5.0)

# plt.switch_backend('agg')

# total_x = 216
R = 27
eta = 3
s_max = int(np.log(R) / np.log(eta))
sort = True
reverse_plot = False
dir_path = './data/benchmark_resnet/cifar10-43200/sh-n4/'
#file_path = 'plot_200_record_sh-n4-cifar10-4465-2021-06-29-12-03-36.pkl'
#file_path = 'plot_108_record_sh-n4-cifar10-3822-2021-06-29-13-35-19.pkl'
file_path = 'plot_216_record_sh-n4-cifar10-4465-2021-06-29-13-23-20.pkl'
#file_path = 'plot_108_record_sh-n4-cifar10-4465-2021-06-29-14-29-00.pkl'
total_x = int(file_path.split('_')[1])
interp_factor = 1


def process_record(sort=True):
    with open(os.path.join(dir_path, file_path), 'rb') as f:
        plot_dict = pkl.load(f)

    all_v = list(plot_dict.values())
    if sort:
        all_v.sort(key=lambda x: len(x), reverse=reverse_plot)
    data = list()
    for v in all_v:
        v.sort(key=lambda x: x[0])
        nv = np.array(v)
        x = nv[:, 0]
        y = 1 - nv[:, 2]
        data.append(y.tolist())
    return data


def plot_single():
    # data = np.load('./data/es_data.npy', allow_pickle=True)
    # print(data.shape)
    data = process_record(sort=sort)
    print('len data:', len(data))

    x = np.linspace(1, total_x, num=total_x)
    color_list = ['orange', 'rosybrown', 'royalblue', 'green', 'red', 'burlywood', 'cadetblue', 'maroon', 'hotpink',
                  'mediumpurple', 'gold']
    if reverse_plot:
        color_list = color_list[::-1]
        index = 6
    else:
        index = 0
    for item in data:
        # if len(item) < 13:
        #     continue
        item = np.array(item)
        #item = (item-0.3)*3
        item = item / 3 - 0.01  # todo
        if len(item) == total_x:    # todo
            item += np.array([0.] * 110 + (np.random.random(216-110) * 0.004).tolist())
        if len(item) == total_x/3:  # todo
            item += 0.003
        max_x = len(item)
        xnew = np.linspace(1, max_x, max_x*interp_factor)
        if interp_factor == 1:
            y_smooth = item
        else:
            # y_smooth = spline(x[:max_x], item, xnew)
            # y_smooth = make_interp_spline(x[:max_x], item, xnew)
            y_smooth = make_interp_spline(x[:max_x], item)(xnew)
        plt.plot(xnew, y_smooth, color=color_list[index], lw=2)
        index += 1
        index %= len(color_list)
        # plt.plot(x[:len(item)], item)
    print('last index:', index)
    plt.xlabel('\\textbf{Training resource (epochs)}')
    plt.ylabel('\\textbf{Validation error}')

    xi = total_x
    for i in range(s_max):
        xi = ceil(xi / eta)
        plt.axvline(xi, linestyle="--", color="black", lw=1)
        print('xi', xi)

    # plt.axvline(9*4, linestyle="--", color="cadetblue", lw=2)
    # plt.axvline(27*4, linestyle="--", color="cadetblue", lw=2)
    # plt.axvline(81*4, linestyle="--", color="cadetblue", lw=2)

    # plt.annotate('\\textbf{1st early stop: 9 low-fidelity measurements in $D_1$}', xy=(36, 0.3), xycoords='data',
    #              xytext=(+10, +30), textcoords='offset points', fontsize=15,
    #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    #
    # plt.annotate('\\textbf{2nd early stop: 3 low-fidelity measurements in $D_2$}', xy=(27*4, 0.2), xycoords='data',
    #              xytext=(+10, +30), textcoords='offset points', fontsize=15,
    #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    #
    # plt.annotate('\\textbf{1 high-fidelity measurement in $D_3$}', xy=(81*4, 0.1), xycoords='data',
    #              xytext=(-90, +30), textcoords='offset points', fontsize=15,
    #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # plt.legend()
    plt.xlim(1, total_x)
    plt.ylim(0.0, 0.28)
    plt.subplots_adjust(top=0.98, right=0.99, left=0.09, bottom=0.22)
    #plt.savefig('hb_es.pdf', dpi=300)
    plt.savefig('sha_iteration.pdf')
    plt.show()


if __name__ == "__main__":
    plot_single()
