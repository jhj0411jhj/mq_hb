import os
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../soln-ml/')
from solnml.datasets.utils import load_data
from solnml.components.utils.constants import MULTICLASS_CLS

data_dir = '../soln-ml/'
# datasets = ['mnist_784', 'higgs', 'covertype']
# datasets = ['spambase', ]
# datasets = ['covtype', 'codrna']

new_data_dir = 'datasets'
if not os.path.exists(new_data_dir):
    os.makedirs(new_data_dir)

for dataset in datasets:
    x, y, feature_type = load_data(dataset, data_dir, False, task_type=MULTICLASS_CLS)
    print(dataset, 'loaded')

    # split. train : validate : test = 6 : 2 : 2
    xx, x_test, yy, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(xx, yy, test_size=0.25, stratify=yy, random_state=1)
    print(dataset, 'split')

    # save
    name_x_train = dataset + '-x_train.npy'
    name_x_val = dataset + '-x_val.npy'
    name_x_test = dataset + '-x_test.npy'
    name_y_train = dataset + '-y_train.npy'
    name_y_val = dataset + '-y_val.npy'
    name_y_test = dataset + '-y_test.npy'
    np.save(os.path.join(new_data_dir, name_x_train), x_train)
    np.save(os.path.join(new_data_dir, name_x_val), x_val)
    np.save(os.path.join(new_data_dir, name_x_test), x_test)
    np.save(os.path.join(new_data_dir, name_y_train), y_train)
    np.save(os.path.join(new_data_dir, name_y_val), y_val)
    np.save(os.path.join(new_data_dir, name_y_test), y_test)
    print(dataset, 'finished')
