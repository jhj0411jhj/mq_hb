"""
example cmdline:

python test/resnet/test_resnet_default.py

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
from resnet_obj import dl_holdout_validation, get_score

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
    # load test
    image_data.set_test_path(data_dir)
    image_data.load_test_data(data_transforms['val'])
    start_time = time.time()

    config_dict = config.get_dictionary().copy()

    estimator = get_estimator(config_dict, max_epoch, device=device, resnet_depth=32)

    epoch_ratio = 1

    estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio)

    try:
        score = dl_holdout_validation(estimator, scorer, image_data, random_state=1)
        test_score = get_score(estimator, scorer, image_data, random_state=1, run_test=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        score = -MAXINT
        test_score = -MAXINT
    train_time = time.time() - start_time
    print('Evaluation | Score: %.4f | Time cost: %.2f seconds' %
          (scorer._sign * score,
           time.time() - start_time))
    print(str(config))

    perf = -score
    test_perf = -test_score
    evals_result = [estimator.train_perf_list, estimator.val_perf_list]
    return perf, test_perf, evals_result, train_time


cs = ResNetClassifier.get_hyperparameter_search_space()

conf_dict = dict(   # see Deep Residual Learning for Image Recognition (CVPR 2016) Kaiming He. setting for cifar10
    optimizer='SGD',
    sgd_learning_rate=0.1,
    sgd_momentum=0.9,
    nesterov='False',
    batch_size=128,
    lr_decay=1e-1,
    weight_decay=1e-4,
    epoch_num=200,
)

from openbox.utils.config_space.space_utils import get_config_from_dict
config = get_config_from_dict(conf_dict, cs)
print(config)

with timeit('%s' % (dataset, )):
    print('=== resnet default param (CVPR2016 Kaiming He)===')
    perf, test_perf, evals_result, train_time = resnet_objective_func_gpu(config)
    print(evals_result)
    print('=== perf(val, test):', perf, test_perf)
    print('=== train time(s):', train_time)
    # print(list(evals_result['validation_0'].values())[0])

    save_item = (config, perf, test_perf, evals_result, train_time)

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    dir_path = 'data/default_test/resnet-%s/' % (dataset, )
    file_name = 'resnet_default-%s-%04d-%s.pkl' % (dataset, seed, timestamp)
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        pass
    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pkl.dump(save_item, f)
    print('save to:', dir_path, file_name)
