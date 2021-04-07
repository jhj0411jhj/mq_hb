"""
example cmdline:

python test/manual_split_data.py --dataset kuaishou2

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
from ks.ks_utils import load_ks_df1, load_ks_df2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)

args = parser.parse_args()
dataset = args.dataset
kwargs = {}


# ===== load dataset =====

t0 = time.time()
from dask_ml.model_selection import train_test_split as dask_train_test_split
from sklearn.model_selection import train_test_split as sk_train_test_split

if dataset == 'kuaishou1':
    load_df_func = load_ks_df1
elif dataset == 'kuaishou2':
    load_df_func = load_ks_df2
else:
    raise ValueError('Unknown ks dataset: %s.' % (dataset,))

df_X, df_y = load_df_func(**kwargs)
print(type(df_X))


# ===== add index =====

assert 'index' not in df_X
df_X['index'] = range(df_X.shape[0])    # fix bug: dask DataFrame reset_index() may generate repeat index


# ===== split data =====

# split. train : validate : test = 6 : 2 : 2
# auto stratify?
# xx, x_test, yy, y_test = dask_train_test_split(df_X, df_y, test_size=0.2, random_state=1)
# x_train, x_val, y_train, y_val = dask_train_test_split(xx, yy, test_size=0.25, random_state=1)

xx, x_test, yy, y_test = sk_train_test_split(df_X, df_y, test_size=0.2, random_state=1, stratify=df_y)
x_train, x_val, y_train, y_val = sk_train_test_split(xx, yy, test_size=0.25, random_state=1, stratify=yy)

print(dataset, 'split', x_train.shape[0], x_val.shape[0], x_test.shape[0])

n_train = np.sum(y_train.values)
n_val = np.sum(y_val.values)
n_test = np.sum(y_test.values)
n_all = np.sum(df_y.values)
print('label value sum=', n_train, n_val, n_test, n_all, n_test / n_all, n_val / (n_train+n_val))
t1 = time.time()
print('===== dataset %s loaded. time=%.2fs' % (dataset, t1-t0), flush=True)


# ===== save split index =====

# assert 'index' not in df_X

# idx_train = x_train.reset_index()['index']
# idx_val = x_val.reset_index()['index']
# idx_test = x_test.reset_index()['index']
idx_train = x_train['index']
idx_val = x_val['index']
idx_test = x_test['index']

# check
idx = list(idx_train) + list(idx_val) + list(idx_test)
print('idx len: ',
      len(idx_train), len(set(idx_train)),
      len(idx_val), len(set(idx_val)),
      len(idx_test), len(set(idx_test)),
      len(idx), len(set(idx)),)
for i in range(3):
    print('idx count =', list(idx_train).count(i), list(idx_val).count(i), list(idx_test).count(i))
try:
    assert len(idx_train) == len(set(idx_train))
    assert len(idx_val) == len(set(idx_val))
    assert len(idx_test) == len(set(idx_test))
    assert len(idx) == len(set(idx))
except AssertionError as e:
    print('===== error index!!! =====')
    idx = df_X['index']
    print('all idx len: ', len(idx), len(set(idx)))
    for i in range(3):
        print('all idx count =', list(idx_train).count(i), list(idx_val).count(i), list(idx_test).count(i))
    raise

save_item = (idx_train, idx_val, idx_test)
dir_path = 'datasets/'
file_name = '%s_index.pkl' % (dataset,)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
with open(os.path.join(dir_path, file_name), 'wb') as f:
    pkl.dump(save_item, f)
print(dir_path, file_name, 'saved!', flush=True)

