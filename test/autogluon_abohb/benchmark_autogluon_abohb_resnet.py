"""
example cmdline:

python test/autogluon_abohb/benchmark_autogluon_abohb_resnet.py --R 27 --reduction_factor 3 --brackets 4 --num_gpus 1 --n_workers 4 --timeout 600 --rep 1 --start_id 0

"""

import autogluon.core as ag
import os
import sys
import time
import logging
import yaml
import argparse
import numpy as np
import pickle as pkl
from math import ceil, log
import torch

sys.path.insert(0, ".")
sys.path.insert(1, "./test/resnet")
sys.path.insert(2, "../open-box")    # for dependency
from test.utils import seeds
try:
    from sklearn.metrics.scorer import accuracy_scorer
except ModuleNotFoundError:
    from sklearn.metrics._scorer import accuracy_scorer
    print('from sklearn.metrics._scorer import accuracy_scorer')
from test.resnet.resnet_model import get_estimator
from test.resnet.resnet_util import get_path_by_config, get_transforms
from test.resnet.resnet_dataset import ImageDataset
from test.resnet.resnet_obj import dl_holdout_validation

from openbox.utils.constants import MAXINT

# Constant
max_epoch = 200
scorer = accuracy_scorer
image_size = 32
data_dir = './datasets/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size)


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Runs autogluon to optimize the hyperparameters')
parser.add_argument('--num_trials', default=999999, type=int,
                    help='number of trial tasks. It is enough to either set num_trials or timout.')
parser.add_argument('--timeout', default=60, type=int,
                    help='runtime of autogluon in seconds. It is enough to either set num_trials or timout.')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='number of GPUs available to a given trial.')  # autogluon use this to infer n_workers
parser.add_argument('--num_cpus', type=int, default=2,
                    help='number of CPUs available to a given trial.')
parser.add_argument('--hostfile', type=argparse.FileType('r'))
parser.add_argument('--store_results_period', type=int, default=100,
                    help='If specified, results are stored in intervals of '
                         'this many seconds (they are always stored at '
                         'the end)')
parser.add_argument('--scheduler', type=str, default='hyperband_promotion',
                    choices=['hyperband_stopping', 'hyperband_promotion'],
                    help='Asynchronous scheduler type. In case of doubt leave it to the default')
parser.add_argument('--reduction_factor', type=int, default=3,
                    help='Reduction factor for successive halving')
parser.add_argument('--brackets', type=int, default=4,
                    help='Number of brackets. Setting the number of brackets to 1 means '
                         'that we run effectively successive halving')
parser.add_argument('--min_resource_level', type=int, default=1,
                    help='Minimum resource level (i.e epochs) on which a configuration is evaluated on.')
parser.add_argument('--searcher', type=str, default='bayesopt',
                    choices=['random', 'bayesopt'],
                    help='searcher to sample new configurations')

# parser.add_argument('--dataset', type=str)
parser.add_argument('--R', type=int, default=27)    # remember to set reduction_factor(eta) and brackets(s_max+1)
parser.add_argument('--n_workers', type=int)        # must set for saving results, but no use in autogluon
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()

model_name = 'resnet'
algo_name = 'abohb_aws'
dataset = 'cifar10'
R = args.R
eta = args.reduction_factor
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
rep = args.rep
start_id = args.start_id

runtime_limit = args.timeout

print(n_workers, dataset)
print(R)
for para in (R, n_workers):
    assert para is not None


@ag.args(  # config space
    optimizer=ag.space.Categorical('SGD', ),
    sgd_learning_rate=ag.space.Real(lower=1e-3, upper=1e-1, default=1e-1, log=True),
    sgd_momentum=ag.space.Real(lower=0.5, upper=0.99, default=0.9, log=False),
    nesterov=ag.space.Categorical('True', 'False'),
    batch_size=ag.space.Categorical(64, 128, 256),
    lr_decay=ag.space.Real(lower=1e-2, upper=2e-1, default=1e-1),
    weight_decay=ag.space.Real(lower=1e-5, upper=1e-2, default=1e-4, log=True),
    epoch_num=ag.space.Categorical(200, ),
    epochs=R,   # max resource
)
def objective_function(args, reporter):
    model_dir = os.path.join('./data/resnet_save_models', method_id)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except FileExistsError:
        pass

    s_max = int(log(args.epochs) / log(eta))
    iterate_r = [int(r) for r in np.logspace(0, s_max, s_max + 1, base=eta)]

    # Hyperparameters to be optimized
    config_dict = dict()
    config_dict['optimizer'] = args.optimizer
    config_dict['sgd_learning_rate'] = args.sgd_learning_rate
    config_dict['sgd_momentum'] = args.sgd_momentum
    config_dict['nesterov'] = args.nesterov
    config_dict['batch_size'] = args.batch_size
    config_dict['lr_decay'] = args.lr_decay
    config_dict['weight_decay'] = args.weight_decay
    config_dict['epoch_num'] = args.epoch_num

    device = 'cuda'
    data_transforms = get_transforms(image_size=image_size)
    image_data.load_data(data_transforms['train'], data_transforms['val'])
    estimator = get_estimator(config_dict, max_epoch, device=device, resnet_depth=32)

    total_resource = args.epochs
    for idx, n_resource in enumerate(iterate_r):
        t_eval_start = time.time()

        epoch_ratio = float(n_resource) / float(total_resource)

        config_model_path = os.path.join(model_dir,
                                         'tmp_' + get_path_by_config(config_dict, is_dict=True) + '_%d' % int(n_resource / eta) + '.pt')
        save_path = os.path.join(model_dir,
                                 'tmp_' + get_path_by_config(config_dict, is_dict=True) + '_%d' % int(n_resource) + '.pt')

        # Continue training
        if idx > 0:
            if not os.path.exists(config_model_path):
                raise ValueError('not initial_run but config_model_path not exists. check if exists duplicated configs '
                                 'and saved model were removed.')
            estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio) - ceil(
                estimator.max_epoch * epoch_ratio / eta)
            estimator.load_path = config_model_path
        else:
            estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio)
        print(estimator.epoch_num)

        try:
            score = dl_holdout_validation(estimator, scorer, image_data, random_state=1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            score = -MAXINT

        test_perf = 12345.6     # todo
        t_eval_end = time.time()
        eval_time = t_eval_end - t_eval_start
        print('Evaluation | Score: %.4f | Time cost: %.2f seconds | config: %s' %
              (scorer._sign * score, eval_time, config_dict))
        print('Report %d/%d: config=%s, score=%f, test_perf=%f.'
              % (n_resource, total_resource, config_dict, score, test_perf), flush=True)

        # remove last model
        if idx > 0:
            try:
                os.remove(config_model_path)
            except Exception as e:
                print('unexpected exception!')
                import traceback
                traceback.print_exc()

        reporter(
            epoch=n_resource,
            performance=float(score),    # Caution: maximized
            eval_time=eval_time,
            time_step=t_eval_end,
            test_perf=test_perf,
            **config_dict,
        )

        # Save low-resource models
        if np.isfinite(score) and epoch_ratio != 1.0:
            state = {'model': estimator.model.state_dict(),
                     'optimizer': estimator.optimizer_.state_dict(),
                     'scheduler': estimator.scheduler.state_dict(),
                     'cur_epoch_num': estimator.cur_epoch_num}
            torch.save(state, save_path)


def evaluate(hyperband_type, method_id, seed, dir_path):
    def callback(training_history, start_timestamp, config_history, state_dict):
        # This callback function will be executed every time AutoGluon collected some new data.
        # In this example we will parse the training history into a .csv file and save it to disk, such that we can
        # for example plot AutoGluon's performance during the optimization process.
        # If you don't care about analyzing performance of AutoGluon online, you can also ignore this callback function and
        # just save the training history after AutoGluon has finished.
        import pandas as pd
        task_dfs = []

        # this function need to be changed if you return something else than accuracy
        def compute_error(df):
            return -df["performance"]  # Caution: minimize

        def compute_runtime(df, start_timestamp):
            return df["time_step"] - start_timestamp

        for task_id in training_history:
            task_df = pd.DataFrame(training_history[task_id])
            task_df = task_df.assign(task_id=task_id,
                                     runtime=compute_runtime(task_df, start_timestamp),
                                     error=compute_error(task_df),
                                     target_epoch=task_df["epoch"].iloc[-1])
            task_dfs.append(task_df)

        result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)

        # re-order by runtime
        result = result.sort_values(by="runtime")

        # calculate incumbent best -- the cumulative minimum of the error.
        result = result.assign(best=result["error"].cummin())

        csv_path = os.path.join(dir_path, "result_%s.csv" % method_id)
        result.to_csv(csv_path)
        print('result saved to', csv_path)

    scheduler = ag.scheduler.HyperbandScheduler(objective_function,
                                                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                                                # Autogluon runs until it either reaches num_trials or time_out
                                                num_trials=args.num_trials,
                                                time_out=runtime_limit,
                                                # This argument defines the metric that will be maximized.
                                                # Make sure that you report this back in the objective function.
                                                reward_attr='performance',
                                                # The metric along we make scheduling decision. Needs to be also
                                                # reported back to AutoGluon in the objective function.
                                                time_attr='epoch',
                                                brackets=args.brackets,
                                                checkpoint=None,
                                                searcher=args.searcher,  # Defines searcher for new configurations
                                                search_options=dict(random_seed=seed, first_is_default=False),
                                                dist_ip_addrs=dist_ip_addrs,
                                                training_history_callback=callback,
                                                training_history_callback_delta_secs=args.store_results_period,
                                                reduction_factor=args.reduction_factor,
                                                type=hyperband_type,
                                                # defines the minimum resource level for Hyperband,
                                                # i.e the minimum number of epochs
                                                grace_period=args.min_resource_level,
                                                random_seed=seed,
                                                )
    # set config space seed
    scheduler.searcher.gp_searcher.configspace_ext.hp_ranges_ext.config_space.seed(seed)
    scheduler.searcher.gp_searcher.configspace_ext.hp_ranges.config_space.seed(seed)
    scheduler.searcher.gp_searcher.hp_ranges.config_space.seed(seed)
    # run
    scheduler.run()
    scheduler.join_jobs()

    # final training history
    training_history = scheduler.training_history
    print('len of final history: %d' % len(training_history), flush=True)
    return training_history


if __name__ == "__main__":
    # In case you want to run AutoGluon across multiple instances, you need to provide it with a list of
    # the IP addresses of the instances. Here, we assume that the IP addresses are stored in a .yaml file.
    # If you want to run AutoGluon on a single instance just pass None.
    # However, keep in mind that it will still parallelize the optimization process across multiple threads then.
    # If you really want to run it purely sequentially, set the num_cpus equal to the number of VCPUs of the machine.
    dist_ip_addrs = yaml.load(args.hostfile) if args.hostfile is not None else []
    del args.hostfile
    print("Got worker host IP addresses [{}]".format(dist_ip_addrs))

    if args.scheduler == "hyperband_stopping":
        hyperband_type = "stopping"
    elif args.scheduler == "hyperband_promotion":
        hyperband_type = "promotion"
    else:
        raise ValueError(args.scheduler)

    # setup
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        if R != 27:
            method_str = '%s-%d-n%d' % (algo_name, R, n_workers)
        else:
            method_str = '%s-n%d' % (algo_name, n_workers)
        if hyperband_type == 'stopping':
            method_str = 's-' + method_str
            print('using hyperband_stopping.')
        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

        dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model_name, dataset, runtime_limit, method_str)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            pass

        # run
        global_start_time = time.time()
        training_history = evaluate(hyperband_type, method_id, seed, dir_path)

        # save
        file_name = 'history_%s.pkl' % (method_id,)
        save_item = (global_start_time, training_history)
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            pkl.dump(save_item, f)
        print(dir_path, file_name, 'saved!', flush=True)

        if rep > 1:
            sleep_time = min(runtime_limit/5, 1800)
            print('sleep %.2fs now!' % sleep_time, flush=True)
            time.sleep(sleep_time)
