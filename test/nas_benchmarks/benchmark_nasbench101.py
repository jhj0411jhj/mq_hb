"""
example cmdline:

python test/nas_benchmarks/benchmark_nasbench101.py --mths hyperband --R 27 --n_workers 4 --runtime_limit 86400 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import pickle as pkl
from functools import partial

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from test.nas_benchmarks.nasbench101_utils import load_nasbench101, get_nasbench101_configspace, objective_func
from test.nas_benchmarks.simulation_utils import run_in_parallel, run_async
from test.utils import seeds, timeit
from test.benchmark_process_record import remove_partial, get_incumbent
from mq_hb.mq_random_search import mqRandomSearch
from mq_hb.mq_sh import mqSuccessiveHalving
from mq_hb.mq_hb import mqHyperband
from mq_hb.mq_bohb import mqBOHB
from mq_hb.mq_bohb_v0 import mqBOHB_v0
from mq_hb.mq_mfes import mqMFES
from mq_hb.mq_mfes_v4 import mqMFES_v4
from mq_hb.async_mq_sh import async_mqSuccessiveHalving
from mq_hb.async_mq_sh_v0 import async_mqSuccessiveHalving_v0
from mq_hb.async_mq_sh_v2 import async_mqSuccessiveHalving_v2
from mq_hb.async_mq_hb import async_mqHyperband
from mq_hb.async_mq_hb_v0 import async_mqHyperband_v0
from mq_hb.async_mq_hb_v2 import async_mqHyperband_v2
from mq_hb.async_mq_weight_hb import async_mqWeightHyperband
from mq_hb.async_mq_bohb import async_mqBOHB
from mq_hb.async_mq_mfes_v3 import async_mqMFES_v3
from mq_hb.async_mq_mfes_v6 import async_mqMFES_v6
from mq_hb.async_mq_mfes_v12 import async_mqMFES_v12
from mq_hb.async_mq_mfes_v13 import async_mqMFES_v13
from mq_hb.async_mq_mfes_v14 import async_mqMFES_v14
from mq_hb.async_mq_mfes_v15 import async_mqMFES_v15

mth_dict = dict(
    random=(mqRandomSearch, 'sync'),
    sh=(mqSuccessiveHalving, 'sync'),
    hyperband=(mqHyperband, 'sync'),
    bohb=(mqBOHB, 'sync'),
    bohbv0=(mqBOHB_v0, 'sync'),
    mfes=(mqMFES, 'sync'),
    mfesv4=(mqMFES_v4, 'sync'),
    asha=(async_mqSuccessiveHalving, 'async'),
    ashav0=(async_mqSuccessiveHalving_v0, 'async'),
    ashav2=(async_mqSuccessiveHalving_v2, 'async'),
    ahb=(async_mqHyperband, 'async'),
    ahbv0=(async_mqHyperband_v0, 'async'),
    ahbv2=(async_mqHyperband_v2, 'async'),
    aweighthb=(async_mqWeightHyperband, 'async'),
    abohb=(async_mqBOHB, 'async'),
    amfesv3=(async_mqMFES_v3, 'async'),
    amfesv6=(async_mqMFES_v6, 'async'),
    amfesv12=(async_mqMFES_v12, 'async'),
    amfesv13=(async_mqMFES_v13, 'async'),
    amfesv14=(async_mqMFES_v14, 'async'),
    amfesv15=(async_mqMFES_v15, 'async'),
)

parser = argparse.ArgumentParser()
parser.add_argument('--mths', type=str, default='hyperband')
# parser.add_argument('--datasets', type=str, default='cifar10')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int)        # must set
parser.add_argument('--runtime_limit', type=int, default=86400)
parser.add_argument('--time_limit_per_trial', type=int, default=999999)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--data_path', type=str, default='../nas_data/nasbench_full.tfrecord')

args = parser.parse_args()
mths = args.mths.split(',')
print("mths:", mths)
# test_datasets = args.datasets.split(',')
# print("datasets num=", len(test_datasets))
R = args.R
eta = args.eta
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
runtime_limit = args.runtime_limit
time_limit_per_trial = args.time_limit_per_trial
rep = args.rep
start_id = args.start_id
data_path = args.data_path

print(n_workers)
print(R, eta)
print(runtime_limit)
for para in (R, eta, n_workers, runtime_limit):
    assert para is not None

for algo_name in mths:
    assert algo_name in mth_dict.keys()


def evaluate_simulation(algo_class, algo_kwargs, method_id, n_workers, seed, parallel_strategy):
    print(method_id, n_workers, seed)

    assert parallel_strategy in ['sync', 'async']

    if parallel_strategy == 'sync':
        algo_class.run_in_parallel = run_in_parallel
    elif parallel_strategy == 'async':
        algo_class.run = run_async
    while True:
        try:
            port = 13579 + np.random.RandomState(int(time.time() * 10000 % 10000)).randint(2000)
            print('port =', port)
            algo = algo_class(
                objective_func=objective_function,  # must set for simulation
                config_space=cs,
                R=R,
                eta=eta,
                random_state=seed,
                method_id=method_id,
                restart_needed=True,
                time_limit_per_trial=time_limit_per_trial,
                runtime_limit=runtime_limit,
                port=port,
                **algo_kwargs,
            )
        except EOFError:
            print('EOFError: try next port.')
        else:
            break
    algo.n_workers = n_workers  # must set for simulation
    algo.run()
    try:
        algo.logger.info('===== bracket status: %s' % algo.get_bracket_status(algo.bracket))
    except Exception as e:
        pass
    try:
        algo.logger.info('===== brackets status: %s' % algo.get_brackets_status(algo.brackets))
    except Exception as e:
        pass
    return algo.recorder


with timeit('load nasbench101'):
    model_name = 'nasbench101'
    dataset = 'cifar10'
    cs = get_nasbench101_configspace()
    nasbench = load_nasbench101(path=data_path)
    objective_function = partial(objective_func, total_resource=R, eta=eta, nasbench=nasbench)

with timeit('all'):
    for algo_name in mths:
        with timeit('%s %d %d' % (algo_name, start_id, rep)):
            algo_class, parallel_strategy = mth_dict[algo_name]
            print('===== start eval %s: rep=%d, runtime_limit=%d, time_limit_per_trial=%d'
                  % (dataset, rep, runtime_limit, time_limit_per_trial))
            for i in range(start_id, start_id + rep):
                seed = seeds[i]

                timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                method_str = '%s-n%d' % (algo_name, n_workers)
                method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

                algo_kwargs = dict()
                with timeit('%d %s' % (i, method_id)):
                    recorder = evaluate_simulation(
                        algo_class, algo_kwargs, method_id, n_workers, seed, parallel_strategy
                    )

                dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model_name, dataset, runtime_limit, method_str)
                file_name = 'record_%s.pkl' % (method_id,)
                try:
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                except FileExistsError:
                    pass
                with open(os.path.join(dir_path, file_name), 'wb') as f:
                    pkl.dump(recorder, f)
                print(dir_path, file_name, 'saved!', flush=True)

            try:
                mths = [method_str]
                remove_partial(model_name, dataset, mths, runtime_limit, R)
                get_incumbent(model_name, dataset, mths, runtime_limit)
            except Exception as e:
                print('benchmark process record failed: %s' % (traceback.format_exc(),))
