"""
example cmdline:

python test/benchmark_xgb_smac.py --datasets spambase --n_jobs 4 --runtime_limit 60 --rep 1 --start_id 0

"""
import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl
from sklearn.metrics import balanced_accuracy_score

sys.path.append(".")
sys.path.insert(0, "../lite-bo")    # for dependency
from mq_hb.xgb_model import XGBoost
from utils import load_data, setup_exp, check_datasets, seeds

# default_datasets = 'mnist_784,higgs,covertype'
default_datasets = 'covtype,codrna'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--num_iter', type=int, default=10000)
parser.add_argument('--runtime_limit', type=int, default=60)
parser.add_argument('--time_limit_per_trial', type=int, default=600)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))

n_jobs = args.n_jobs                                # changed according to dataset
num_iter = args.num_iter
runtime_limit = args.runtime_limit                  # changed according to dataset
time_limit_per_trial = args.time_limit_per_trial    # changed according to dataset

rep = args.rep
start_id = args.start_id

print(n_jobs, test_datasets)
print(runtime_limit)
for para in (n_jobs, runtime_limit):
    assert para is not None


def evaluate(method_id, dataset, seed):
    print(method_id, dataset, seed)

    def objective_func(config):
        params = config.get_dictionary()
        model = XGBoost(**params, n_jobs=n_jobs, seed=seed)
        model.fit(x_train, y_train)
        # evaluate on validation data
        y_pred = model.predict(x_val)
        perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
        return perf

    cs = XGBoost.get_cs()
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)

    from smac.scenario.scenario import Scenario
    from smac.facade.smac_facade import SMAC
    from smac_modified import RunHistory_modified   # use modified RunHistory to save record
    # Scenario object
    scenario = Scenario({"run_obj": "quality",
                         "runcount_limit": num_iter,
                         "cs": cs,
                         "wallclock_limit": runtime_limit,
                         "cutoff_time": time_limit_per_trial,
                         # "deterministic": "true",
                         })
    runhistory = RunHistory_modified(None)  # aggregate_func handled by smac_facade.SMAC
    smac = SMAC(scenario=scenario, runhistory=runhistory,
                tae_runner=objective_func, run_id=seed,  # set run_id for smac output_dir
                rng=np.random.RandomState(seed))
    smac.optimize()
    # keys = [k.config_id for k in smac.runhistory.data.keys()]
    # perfs = [v.cost for v in smac.runhistory.data.values()]
    recorder = smac.runhistory.exp_recorder
    return recorder


check_datasets(test_datasets)
for dataset in test_datasets:
    # setup
    n_jobs, runtime_limit, time_limit_per_trial = setup_exp(dataset, n_jobs, runtime_limit, time_limit_per_trial)
    print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
          % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        method_str = 'smac'
        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

        recorder = evaluate(method_id, dataset, seed)

        dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, runtime_limit, method_str)
        file_name = 'record_%s.pkl' % (method_id,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            pkl.dump(recorder, f)
        print(dir_path, file_name, 'saved!', flush=True)
