"""
example cmdline:

python test/benchmark_plot_scatter.py --dataset codrna --time 3600 --mth smac --run_id 0

"""
import argparse
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import seeds

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=int, default=3600)
parser.add_argument('--dataset', type=str, default='codrna')
parser.add_argument('--mth', type=str, default='smac')
parser.add_argument('--run_id', type=int, default=0)

args = parser.parse_args()
max_time = args.time
dataset = args.dataset
mth = args.mth
seed = seeds[args.run_id]

recorder = None
dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, max_time, mth)
for file in os.listdir(dir_path):
    if file.startswith('new_record_%s-%s-%d' % (mth, dataset, seed)) and file.endswith('.pkl'):
        with open(os.path.join(dir_path, file), 'rb') as f:
            save_item = pkl.load(f)
            recorder = save_item
        break

configs = [record['configuration'].get_dictionary() for record in recorder]
perfs = [-record['return_info']['loss'] for record in recorder]     # validation perfs
for hp in (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
):
    x = [conf[hp] for conf in configs]

    title = '%s-%s' % (dataset, mth)
    x_label = hp
    y_label = "Balanced Accuracy Score"
    # if hp == "learning_rate":
    #     x = np.log10(x)
    #     x_label = 'log ' + hp

    plt.scatter(x, perfs)
    # plt.ylim(0.59, 0.675)

    # show plot
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.tight_layout()
    plt.show()
