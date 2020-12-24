import argparse
import numpy as np

import sys
sys.path.append('.')
sys.path.insert(0, '../lite-bo')    # for dependency
from litebo.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from mq_hb.mq_hb import mqHyperband
from mq_hb.mq_mf_worker import mqmfWorker


parser = argparse.ArgumentParser()
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--R', type=int, default=81)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int)

args = parser.parse_args()
role = args.role
ip = args.ip
port = args.port
R = args.R
eta = args.eta
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
print(role, ip, port, n_workers)
print(R, eta)
for para in (role, ip, port, R, eta, n_workers):
    assert para is not None


def get_cs():
    cs = ConfigurationSpace()
    n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=100)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 99, default_value=50)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    cs.add_hyperparameters([n_estimators, num_leaves, learning_rate])
    return cs


def mf_objective_func(config: dict, n_resource):
    uid = config.pop('uid', 1)
    reference = config.pop('reference', None)
    need_lc = config.pop('need_lc', None)
    method_name = config.pop('method_name', None)
    print('objective extra info in config:', uid, reference, need_lc, method_name)

    # todo sample data
    def sample_data(n_resource, total_resource=R):
        print('sample data:', n_resource, total_resource)
        return None
    data = sample_data(n_resource)

    # x, y = load_data()
    # model = LightGBM(**params)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # return 1 - balanced_accuracy_score(y_test, y_pred)

    n_estimators = config['n_estimators']
    num_leaves = config['num_leaves']
    learning_rate = config['learning_rate']
    perf = n_estimators + num_leaves + learning_rate + np.random.rand()/10000

    result = dict(
        objective_value=perf,
        early_stop=False,   # for deep learning
        ref_id=None,
    )
    return result


cs = get_cs()

if role == 'master':
    random_state = 123
    dataset = 'nodata'
    method_id = 'hyperband-%s-n%d-%d' % (dataset, n_workers, random_state)
    hyperband = mqHyperband(None, cs, R, eta=eta, method_id=method_id, ip=ip, port=port,
                            restart_needed=True, time_limit_per_trial=600)
    hyperband.num_iter = 1              # set repeat num of hyperband
    hyperband.runtime_limit = None      # set total runtime limit
    hyperband.run()
else:
    worker = mqmfWorker(mf_objective_func, ip, port)
    worker.run()
