import sys
from functools import partial
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')
from mq_hb.mq_hb import mqHyperband
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter
from test.nas_benchmarks.simulation_utils import run_in_parallel, run_async


cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('t', 1, 50))
cs.add_hyperparameter(UniformIntegerHyperparameter('v', 0, 100))


def obj_func(config, n_resource, extra_conf, total_resource, eta):
    val = config['v']
    train_time = config['t']
    result = dict(
        objective_value=-val,  # minimize
        elapsed_time=train_time,
    )
    return result


R = 9
eta = 3
num_iter = 2

obj_func = partial(obj_func, total_resource=R, eta=eta)

algo_class = mqHyperband
algo_class.run_in_parallel = run_in_parallel
algo = algo_class(
    objective_func=obj_func,    # must set for simulation
    config_space=cs,
    R=R, eta=eta,
    num_iter=num_iter,
    restart_needed=True,
    runtime_limit=None,
)
algo.n_workers = 4  # must set for simulation
algo.run()
