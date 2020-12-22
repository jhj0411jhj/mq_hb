import argparse
import numpy as np
from mq_hb import mqHyperband
from mq_mf_worker import mqmfWorker

import sys
sys.path.insert(0, '../lite-bo')
from litebo.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter


parser = argparse.ArgumentParser()
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)

args = parser.parse_args()
role = args.role
ip = args.ip
port = args.port
R = args.R
eta = args.eta


def get_cs():
    cs = ConfigurationSpace()
    n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=100)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 99, default_value=50)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    cs.add_hyperparameters([n_estimators, num_leaves, learning_rate])
    return cs


def mf_objective_func(config: dict, n_iter):
    uid = config.pop('uid', 1)
    reference = config.pop('reference', None)
    need_lc = config.pop('need_lc', None)
    method_name = config.pop('method_name', None)
    print('objective extra info in config:', uid, reference, need_lc, method_name)

    # todo sample data
    def sample_data(n_iter, total_iter):
        print('sample data:', n_iter, total_iter)
        return None
    data = sample_data(n_iter, R)

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
    return perf


cs = get_cs()

if role == 'master':
    hyperband = mqHyperband(None, cs, R, eta=eta, ip=ip, port=port)
    hyperband.run()
else:
    worker = mqmfWorker(mf_objective_func, ip, port)
    worker.run()
