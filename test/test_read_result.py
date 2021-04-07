"""
test read saved data. see mqBaseFacade.save_intermediate_statistics()
"""
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

data_dir = './data/'

method = 'hyperband-n1'
dataset = 'spambase'
seed = 123
max_iter = 27

# read time perf npy
file_name = '%s-%s-%d.npy' % (method, dataset, seed)
file_path = os.path.join(data_dir, file_name)
xy = np.load(file_path)
x = xy[0]
y = xy[1]
plt.plot(x, y)
plt.xlabel('Time elapsed (sec)')
plt.ylabel('Validation error')
plt.show()

# read config pkl
file_name = 'config_%s-%s-%d.pkl' % (method, dataset, seed)
file_path = os.path.join(data_dir, file_name)
with open(file_path, 'rb') as f:
    global_incumbent_configuration = pkl.load(f)
    print('=====config')
    print(global_incumbent_configuration)   # Caution: part validated incumbent

# read record pkl
file_name = 'record_%s-%s-%d.pkl' % (method, dataset, seed)
file_path = os.path.join(data_dir, file_name)
with open(file_path, 'rb') as f:
    recorder = pkl.load(f)
    print('=====recorder')
    best_perf = 2 ** 31 - 1
    best_record = None
    for record in recorder:  # trial_id, time_consumed, configuration, n_iteration, return_info, global_time
        if record['n_iteration'] >= max_iter:
            print(record)
        if record['return_info']['loss'] < best_perf:   # Caution: part validated incumbent
            best_perf = record['return_info']['loss']
            best_record = record
    print('=====recorder best (part validated)')
    print(best_record)
