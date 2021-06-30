"""
for plotting curve

example cmdline:

python test/resnet/benchmark_resnet_runplot.py --max_epoch 200 --mth sh --R 27 --n_workers 2 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl
from collections import OrderedDict

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency

import traceback
from functools import partial
from multiprocessing import Process, Manager

from mq_hb import mth_dict
from mq_hb.mq_mf_worker_gpu import mqmfWorker_gpu
from mq_hb.async_mq_mf_worker_gpu import async_mqmfWorker_gpu
from test.utils import setup_exp, seeds
from resnet_model import ResNet32Classifier

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

# Constant
#max_epoch = 54      # todo 200->54
scorer = accuracy_scorer
image_size = 32
data_dir = './datasets/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size)



parser = argparse.ArgumentParser()
parser.add_argument('--mth', type=str, default='sh')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_jobs', type=int, default=4)

parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--n_workers', type=int)        # must set

parser.add_argument('--runtime_limit', type=int, default=43200)
parser.add_argument('--time_limit_per_trial', type=int, default=999999)

parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

parser.add_argument('--max_epoch', type=int, default=200)

args = parser.parse_args()
algo_name = args.mth
dataset = args.dataset
R = args.R
eta = args.eta
n_jobs = args.n_jobs                                # changed according to dataset

ip = args.ip
port = args.port
n_workers = args.n_workers  # Caution: must set for saving result to different dirs

runtime_limit = args.runtime_limit                  # changed according to dataset
time_limit_per_trial = args.time_limit_per_trial    # changed according to dataset

rep = args.rep
start_id = args.start_id

max_epoch = args.max_epoch

print(ip, port, n_jobs, n_workers, dataset)
print(R, eta)
print(runtime_limit)
for para in (ip, port, n_jobs, R, eta, n_workers, runtime_limit):
    assert para is not None

mth_info = mth_dict[algo_name]
if len(mth_info) == 2:
    algo_class, parallel_strategy = mth_info
    algo_kwargs = dict()
elif len(mth_info) == 3:
    algo_class, parallel_strategy, algo_kwargs = mth_info
else:
    raise ValueError('error mth info: %s' % mth_info)

assert dataset == 'cifar10'
model = 'resnet'

def mf_objective_func_gpu(config, n_resource, extra_conf, device, total_resource, run_test=False,
                          model_dir='./data/resnet_save_models/unnamed_trial', eta=3):    # device='cuda' 'cuda:0'
    print('extra_conf:', extra_conf)
    initial_run = extra_conf['initial_run']
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except FileExistsError:
        pass

    data_transforms = get_transforms(image_size=image_size)
    image_data.load_data(data_transforms['train'], data_transforms['val'])
    start_time = time.time()

    config_dict = config.get_dictionary().copy()

    estimator = get_estimator(config_dict, max_epoch, device=device)

    epoch_ratio = float(n_resource) / float(total_resource)

    config_model_path = os.path.join(model_dir,
                                     'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource / eta) + '.pt')
    save_path = os.path.join(model_dir,
                             'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource) + '.pt')

    # Continue training if initial_run=False
    if not initial_run:
        if not os.path.exists(config_model_path):
            raise ValueError('not initial_run but config_model_path not exists. check if exists duplicated configs '
                             'and saved model were removed.')
        estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio) - ceil(
            estimator.max_epoch * epoch_ratio / eta)
        estimator.load_path = config_model_path
        print(estimator.epoch_num)
    else:
        estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio)

    try:
        score = dl_holdout_validation(estimator, scorer, image_data, random_state=1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        score = -MAXINT
    print('Evaluation | Score: %.4f | Time cost: %.2f seconds' %
          (scorer._sign * score,
           time.time() - start_time))
    print(str(config))

    # Save low-resource models
    if np.isfinite(score) and epoch_ratio != 1.0:
        state = {'model': estimator.model.state_dict(),
                 'optimizer': estimator.optimizer_.state_dict(),
                 'scheduler': estimator.scheduler.state_dict(),
                 'cur_epoch_num': estimator.cur_epoch_num}
        torch.save(state, save_path)

    try:
        if epoch_ratio == 1:
            s_max = int(log(total_resource) / log(eta))
            for i in range(0, s_max + 1):
                if os.path.exists(os.path.join(model_dir,
                                               'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt')):
                    os.remove(os.path.join(model_dir,
                                           'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt'))
    except Exception as e:
        print('unexpected exception!')
        import traceback
        traceback.print_exc()

    result = dict(
        objective_value=-score,
        test_perf=estimator.val_perf_list,
    )
    return result

def evaluate_parallel(algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
                      parallel_strategy, n_jobs, R, eta=3, run_test=True,
                      dir_path=None, file_name=None):
    # dataset / n_jobs are ignored
    assert dir_path is not None
    assert file_name is not None

    print(method_id, n_workers, dataset, seed)
    if port == 0:
        port = 13579 + np.random.RandomState(int(time.time() * 10000 % 10000)).randint(2000)
    print('ip=', ip, 'port=', port)
    assert parallel_strategy in ['sync', 'async']

    model_dir = os.path.join('./data/resnet_save_models', method_id)
    objective_function_gpu = partial(mf_objective_func_gpu, total_resource=R, run_test=run_test,
                                     model_dir=model_dir, eta=eta)

    def master_run(return_list, algo_class, algo_kwargs):
        algo_kwargs['ip'] = ''
        algo_kwargs['port'] = port
        algo = algo_class(**algo_kwargs)

        # tmp_path = os.path.join(dir_path, 'tmp')
        # algo.set_save_intermediate_record(tmp_path, file_name)

        algo.run()
        try:
            algo.logger.info('===== bracket status: %s' % algo.get_bracket_status(algo.bracket))
        except Exception as e:
            pass
        try:
            algo.logger.info('===== brackets status: %s' % algo.get_brackets_status(algo.brackets))
        except Exception as e:
            pass
        return_list.extend(algo.recorder)  # send to return list

    def worker_run(i):
        device = 'cuda:%d' % i  # gpu
        if parallel_strategy == 'sync':
            worker = mqmfWorker_gpu(objective_function_gpu, device, ip, port, no_time_limit=True)
        elif parallel_strategy == 'async':
            worker = async_mqmfWorker_gpu(objective_function_gpu, device, ip, port, no_time_limit=True)
        else:
            raise ValueError('Error parallel_strategy: %s.' % parallel_strategy)
        worker.run()
        print("Worker %d exit." % (i,))

    manager = Manager()
    recorder = manager.list()   # shared list
    master = Process(target=master_run, args=(recorder, algo_class, algo_kwargs))
    master.start()

    time.sleep(10)  # wait for master init
    worker_pool = []
    for i in range(n_workers):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:
        w.kill()

    return list(recorder)   # covert to list


# setup
print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
      % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
for i in range(start_id, start_id + rep):
    seed = seeds[i]

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    method_str = '%s-n%d' % (algo_name, n_workers)
    method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

    algo_kwargs = dict()
    algo_kwargs['R'] = R
    algo_kwargs['eta'] = eta
    algo_kwargs['restart_needed'] = True

    algo_kwargs['num_iter'] = 1
    print('num_iter=1')

    # ip, port are filled in evaluate_parallel()
    algo_kwargs['objective_func'] = None
    algo_kwargs['config_space'] = ResNet32Classifier.get_hyperparameter_search_space()
    algo_kwargs['random_state'] = seed
    algo_kwargs['method_id'] = method_id
    algo_kwargs['runtime_limit'] = runtime_limit
    algo_kwargs['time_limit_per_trial'] = time_limit_per_trial

    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, method_str)
    file_name = 'record_%s.pkl' % (method_id,)

    recorder = evaluate_parallel(
        algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
        parallel_strategy, n_jobs, R, eta=eta, run_test=False,
        dir_path=dir_path, file_name=file_name,
    )

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        pass
    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pkl.dump(recorder, f)
    print(dir_path, file_name, 'saved!', flush=True)

    # process
    plot_dict = OrderedDict()
    for record in recorder:
        config = record['configuration']
        val_perf_list = record['return_info']['test_perf']  # list of [epoch, val_avg_loss, val_avg_acc]
        if config not in plot_dict.keys():
            plot_dict[config] = list()
        plot_dict[config].extend(val_perf_list)
        plot_dict[config].sort(key=lambda x: x[0])

    plot_file_name = 'plot_%d_%s' % (max_epoch, file_name)
    with open(os.path.join(dir_path, plot_file_name), 'wb') as f:
        pkl.dump(plot_dict, f)
    print(dir_path, plot_file_name, 'saved!', flush=True)

