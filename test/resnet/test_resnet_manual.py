"""
example cmdline:

python test/default_test/resnet_manual.py

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

from openbox.utils.constants import MAXINT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')

args = parser.parse_args()
dataset = args.dataset

# Constant
max_epoch = 200
scorer = accuracy_scorer
image_size = 32
data_dir = './datasets/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size)

seed = 47  # no use


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


cs = ResNetClassifier.get_hyperparameter_search_space()
config = cs.get_default_configuration()   # todo

with timeit('%s' % (dataset, )):
    print('=== resnet manual param (my default)===')
    perf, test_perf, evals_result, train_time = resnet_objective_func_gpu(config)
    print(evals_result)
    print('=== perf(val, test):', perf, test_perf)
    print('=== train time(s):', train_time)
    # print(list(evals_result['validation_0'].values())[0])

    save_item = (config, perf, test_perf, evals_result, train_time)

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    dir_path = 'data/default_test/resnet-%s/' % (dataset, )
    file_name = 'resnet_manual-%s-%04d-%s.pkl' % (dataset, seed, timestamp)
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        pass
    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pkl.dump(save_item, f)
    print('save to:', dir_path, file_name)
