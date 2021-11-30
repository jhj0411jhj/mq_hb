"""
example cmdline:

python test/default_test/resnet_best.py --mth amfesv19-n4 --rep 5 --start_id 0

"""

import os
import time
import argparse
import numpy as np
import pickle as pkl

import sys
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')    # for dependency
from test.utils import timeit
from resnet_model import ResNetClassifier

import torch
from math import ceil, log
try:
    from sklearn.metrics.scorer import accuracy_scorer
except ModuleNotFoundError:
    from sklearn.metrics._scorer import accuracy_scorer
    print('from sklearn.metrics._scorer import accuracy_scorer')
from resnet_model import get_estimator
from resnet_util import get_path_by_config, get_transforms
from resnet_dataset import ImageDataset
from resnet_obj import dl_holdout_validation
from test.utils import seeds

from openbox.utils.constants import MAXINT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--mth', type=str, default='amfesv19-n4')
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--runtime_limit', type=int, default=172800)

args = parser.parse_args()
dataset = args.dataset
mth = args.mth
rep = args.rep
start_id = args.start_id
runtime_limit = args.runtime_limit

# Constant
max_epoch = 200
scorer = accuracy_scorer
image_size = 32
data_dir = './datasets/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size)


def resnet_objective_func_gpu(config, device='cuda'):    # device='cuda' 'cuda:0'

    data_transforms = get_transforms(image_size=image_size)
    image_data.load_data(data_transforms['train'], data_transforms['val'])
    start_time = time.time()

    config_dict = config.get_dictionary().copy()

    estimator = get_estimator(config_dict, max_epoch, device=device, resnet_depth=32)

    epoch_ratio = 1

    estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio)

    try:
        score = dl_holdout_validation(estimator, scorer, image_data, random_state=1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        score = -MAXINT
    train_time = time.time() - start_time
    print('Evaluation | Score: %.4f | Time cost: %.2f seconds' %
          (scorer._sign * score,
           time.time() - start_time))
    print(str(config))

    perf = -score
    test_perf = None
    evals_result = [estimator.train_perf_list, estimator.val_perf_list]
    return perf, test_perf, evals_result, train_time


print('===== start test %s %s: rep=%d' % (mth, dataset, rep, ))
for i in range(start_id, start_id + rep):
    seed = seeds[i]

    dir_path = 'data/benchmark_resnet/%s-%d/%s/' % (dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('incumbent_new_record_%s-%s-%d-' % (mth, dataset, seed)) \
                and file.endswith('.pkl'):
            # load config
            with open(os.path.join(dir_path, file), 'rb') as f:
                record = pkl.load(f)
            print(dataset, mth, seed, 'loaded!', record, flush=True)

            # run test
            config = record['configuration']
            print('=== resnet best param ===')
            perf, test_perf, evals_result, train_time = resnet_objective_func_gpu(config)
            print(evals_result)
            print('=== perf(val, test):', perf, test_perf)
            print('=== train time(s):', train_time)
            # print(list(evals_result['validation_0'].values())[0])

            save_item = (config, perf, test_perf, evals_result, train_time)

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            save_dir_path = 'data/default_test/resnet-%s/' % (dataset, )
            save_file_name = 'resnet_best-%s-%04d-%s.pkl' % (dataset, seed, timestamp)
            try:
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
            except FileExistsError:
                pass
            with open(os.path.join(save_dir_path, save_file_name), 'wb') as f:
                pkl.dump(save_item, f)
            print('save to:', save_dir_path, save_file_name)
