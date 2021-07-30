import os
import time
import numpy as np
from typing import List

from mq_hb.async_mq_median_stopping import async_mqMedianStopping
from mq_hb.utils import sample_configuration
from mq_hb.utils import minmax_normalization, std_normalization

from mq_hb.surrogate.mf_gp import convert_configurations_to_resource_array, create_resource_gp_model
from mq_hb.acq_maximizer.ei_optimization import mf_RandomSampling

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI


class async_mqMFGP_MedianStopping(async_mqMedianStopping):
    """
    The implementation of Asynchronous MFGP
    + Build model at model_iterations
    + Use multi-fidelity GP
    median stopping variant
    + Report at each n_iteration in [1, R]
    + Decide stopping at n_iteration in stop_iterations
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 stop_iterations: List = None,
                 window_size: int = None,
                 model_iterations: List = None,
                 log_scale_model_iterations=False,
                 rand_prob=0.3,
                 bo_init_num=3,
                 use_botorch_gp=False,
                 random_state=1,
                 method_id='mqAsyncMFGP_median',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 **kwargs):

        super().__init__(objective_func, config_space, R, stop_iterations=stop_iterations, window_size=window_size,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey, **kwargs)

        self.iterate_r = model_iterations
        if self.iterate_r is None:
            self.iterate_r = self.stop_iterations
        self.logger.info('model_iterations: %s.' % self.iterate_r)

        self.log_scale_model_iterations = log_scale_model_iterations

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        self.rng = np.random.RandomState(self.seed)
        types, bounds = get_types(config_space)
        self.num_hps = len(bounds)
        if use_botorch_gp:
            from mq_hb.surrogate.gp_botorch import GaussianProcess_BoTorch
            self.surrogate = GaussianProcess_BoTorch(types, bounds, standardize_y=False)
        else:
            self.surrogate = create_resource_gp_model('gp', config_space, types, bounds, self.rng)
        self.acquisition_function = EI(model=self.surrogate)
        self.acq_optimizer = mf_RandomSampling(self.acquisition_function, config_space,
                                               max_resource=self.R,
                                               n_samples=max(5000, 50 * len(bounds)),
                                               log_scale=self.log_scale_model_iterations)

        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, r in enumerate(self.iterate_r):
            self.target_x[r] = list()
            self.target_y[r] = list()

    def decide_stopping(self, config, perf, n_iteration):
        decision = super().decide_stopping(config, perf, n_iteration)

        self.target_x[n_iteration].append(config)
        self.target_y[n_iteration].append(perf)

        return decision

    def choose_next(self):
        """
        sample a config according to MFES. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()

        # sample config
        excluded_configs = self.all_configs

        if len(self.incumbent_configs) < self.bo_init_num or self.rng.random() < self.rand_prob:
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # BO
            start_time = time.time()
            # train BO surrogate
            train_configs = list()
            train_resources = list()
            train_perfs = list()
            for r in self.iterate_r:
                train_configs.extend(self.target_x[r])
                train_resources.extend([r] * len(self.target_x[r]))
                train_perfs.extend(self.target_y[r])
            X = convert_configurations_to_resource_array(train_configs,
                                                         train_resources,
                                                         max_resource=self.R,
                                                         log_scale=self.log_scale_model_iterations)
            Y = np.array(std_normalization(train_perfs), dtype=np.float64)
            self.surrogate.train(X, Y)
            time1 = time.time()
            # Update surrogate model in acquisition function.
            best_index = int(np.argmin(Y))
            best_config = train_configs[best_index]
            self.acquisition_function.update(model=self.surrogate, eta=Y[best_index],
                                             num_data=X.shape[0])
            # get acq_resource
            acq_resource = 1
            for r in reversed(self.iterate_r):
                if len(self.target_x[r]) >= self.num_hps:
                    acq_resource = r
                    break
            self.logger.info('acq_resource=%d' % (acq_resource,))
            candidates = self.acq_optimizer.maximize(resource=acq_resource, best_config=best_config, batch_size=5000)
            for candidate in candidates:
                if candidate not in excluded_configs:
                    next_config = candidate
                    break
            if next_config is None:
                self.logger.warning('Cannot get a non duplicate configuration from bo candidates. '
                                    'Sample a random one.')
                next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
            time2 = time.time()
            if time2 - start_time > 1:
                self.logger.info('BO training cost %.2fs. acq optimization cost %.2fs.'
                                 % (time1 - start_time, time2 - time1))

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_next_n_iteration(self):
        return 1
