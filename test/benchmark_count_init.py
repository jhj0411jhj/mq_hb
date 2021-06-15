"""
example cmdline:

python test/benchmark_count_init.py --dataset covtype

"""
import argparse
import os
import numpy as np
import pickle as pkl

from utils import setup_exp

#default_mths = 'random-n1,random-n3,smac,hyperband-n1,hyperband-n3,bohb-n1,bohb-n3,mfes-n1,mfes-n3'
default_mths = 'hyperband-n8,bohb-n8,mfes-n8,amfesv0-n8,amfesv3-n8,amfesv6-n8'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--runtime_limit', type=int)    # if you don't want to use default setup
parser.add_argument('--model', type=str, default='xgb')

args = parser.parse_args()
dataset = args.dataset
mths = args.mths.split(',')
model = args.model


print('start', dataset)
# setup
_, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
if args.runtime_limit is not None:
    runtime_limit = args.runtime_limit

plot_list = []
legend_list = []
result = dict()
for mth in mths:
    stats = []
    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                raw_recorder = pkl.load(f)
            # recorder = []
            # for record in raw_recorder:
            #     if record['global_time'] > runtime_limit:
            #         print('abandon record by runtime_limit:', runtime_limit, mth, record)
            #         continue
            #     recorder.append(record)
            # recorder.sort(key=lambda rec: rec['global_time'])
            # print('new recorder len:', mth, len(recorder), len(raw_recorder))
            print('recorder len:', mth, len(raw_recorder))
            count_dict = dict()
            for record in raw_recorder:
                initial_run = record['return_info']['extra_conf']['initial_run']
                if initial_run:
                    n_iteration = record['n_iteration']
                    if n_iteration not in count_dict.keys():
                        count_dict[n_iteration] = 0
                    count_dict[n_iteration] += 1
            print(mth, count_dict)
