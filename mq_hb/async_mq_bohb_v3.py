import os
import time
import numpy as np

from mq_hb.async_mq_hb import async_mqHyperband
from mq_hb.utils import RUNNING, COMPLETED, PROMOTED
from mq_hb.utils import sample_configuration
from mq_hb.utils import minmax_normalization, std_normalization
from mq_hb.surrogate.mf_gp import convert_configurations_to_resource_array, create_resource_gp_model
from mq_hb.acq_maximizer.ei_optimization import mf_RandomSampling

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI


class async_mqBOHB_v3(async_mqHyperband):
    """
    The implementation of A-BOHB (Amazon)
    https://arxiv.org/abs/2003.10865
    using multi-fidelity GP
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 rand_prob=0.3,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqAsyncBOHB_mfgp',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc'):
        super().__init__(objective_func, config_space, R, eta=eta, skip_outer_loop=skip_outer_loop,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        self.rng = np.random.RandomState(self.seed)
        types, bounds = get_types(config_space)
        self.num_hps = len(bounds)
        self.surrogate = create_resource_gp_model('gp', config_space, types, bounds, self.rng)
        self.acquisition_function = EI(model=self.surrogate)
        self.acq_optimizer = mf_RandomSampling(self.acquisition_function, config_space,
                                               max_resource=self.R,
                                               n_samples=max(5000, 50 * len(bounds)))   # todo: median

        self.iterate_r = list()
        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()
        self.n_init_configs = np.array(
            [len(init_iter_list) for init_iter_list in self.hb_bracket_list],
            dtype=np.float64
        )
        self.choose_weights = self.n_init_configs / np.sum(self.n_init_configs)
        self.logger.info('n_init_configs: %s. next_n_iteration choose_weights: %s.'
                         % (self.n_init_configs, self.choose_weights))

    def update_observation(self, config, perf, n_iteration):
        rung_id = self.get_rung_id(self.bracket, n_iteration)

        updated = False
        for job in self.bracket[rung_id]['jobs']:
            _job_status, _config, _perf, _extra_conf = job
            if _config == config:
                assert _job_status == RUNNING
                job[0] = COMPLETED
                job[2] = perf
                updated = True
                break
        assert updated
        # print('=== bracket after update_observation:', self.get_bracket_status(self.bracket))

        n_iteration = int(n_iteration)
        self.target_x[n_iteration].append(config)
        self.target_y[n_iteration].append(perf)

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)

    def choose_next(self):
        """
        sample a config according to BOHB. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        # sample config
        excluded_configs = self.bracket[next_rung_id]['configs']

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
                                                         max_resource=self.R)
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
            self.logger.info('acq_resource=%d' % (acq_resource, ))
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
                                 % (time1-start_time, time2-time1))

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_next_n_iteration(self):
        next_n_iteration = self.rng.choice(self.iterate_r, p=self.choose_weights)
        self.logger.info('random choosing next_n_iteration=%d.' % (next_n_iteration,))
        return next_n_iteration
