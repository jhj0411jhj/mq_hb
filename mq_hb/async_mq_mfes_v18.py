import os
import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from mq_hb.async_mq_hb import async_mqHyperband
from mq_hb.utils import RUNNING, COMPLETED, PROMOTED
from mq_hb.utils import sample_configuration
from mq_hb.utils import minmax_normalization, std_normalization
from mq_hb.surrogate.rf_ensemble import RandomForestEnsemble
from mq_hb.acq_maximizer.ei_optimization import RandomSampling

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer


class async_mqMFES_v18(async_mqHyperband):
    """
    The implementation of Asynchronous MFES (combine ASHA and MFES)
    before fix
    no median
    promotion start threshold: v3
    v6: non_decreasing_weight
    === from v6
    v13: choose n_iteration of new config by unadjusted weight (random choice)
         update_weight when update_observation
    === from v13
    v18: debug version: use_weight_init=False, promotion_start_threshold=0, w_K = 1
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 rand_prob=0.3,
                 use_weight_init=False,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm',
                 fusion_method='idp',
                 power_num=3,
                 non_decreasing_weight=True,
                 random_state=1,
                 method_id='mqAsyncMFES',
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

        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Parameter for weight method `rank_loss_p_norm`.
        self.power_num = power_num
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        if init_weight is None:
            # init_weight = [1. / self.s_max] * self.s_max + [0.]
            init_weight = [0.] * self.s_max + [1.0]  # debug
        assert len(init_weight) == (self.s_max + 1)
        self.logger.info("Initialize weight to %s" % init_weight[:self.s_max + 1])
        types, bounds = get_types(config_space)

        self.surrogate = RandomForestEnsemble(types, bounds, self.s_max, self.eta,
                                              init_weight, self.fusion_method)
        self.acquisition_function = EI(model=self.surrogate)

        self.iterate_id = 0
        self.iterate_r = list()
        self.hist_weights = list()
        self.hist_weights_unadjusted = list()

        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        # BO optimizer settings.
        self.history_container = HistoryContainer(task_id=self.method_name)
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.rng = np.random.RandomState(seed=self.seed)
        # self.acq_optimizer = InterleavedLocalAndRandomSearch(
        #     acquisition_function=self.acquisition_function,
        #     config_space=self.config_space,
        #     rng=self.rng,
        #     max_steps=self.sls_max_steps,
        #     n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
        #     n_sls_iterations=self.n_sls_iterations,
        #     rand_prob=0.0,
        # )
        self.random_configuration_chooser = ChooserProb(prob=rand_prob, rng=self.rng)
        self.random_check_idx = 0

        self.acq_optimizer = RandomSampling(self.acquisition_function, config_space,
                                            n_samples=max(5000, 50 * len(bounds)))

        self.non_decreasing_weight = non_decreasing_weight
        self.use_weight_init = use_weight_init
        self.n_init_configs = np.array(
            [len(init_iter_list) for init_iter_list in self.hb_bracket_list],
            dtype=np.float64
        )

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

        configs_running = list()
        for _config in self.bracket[rung_id]['configs']:
            if _config not in self.target_x[n_iteration]:
                configs_running.append(_config)
        value_imputed = np.median(self.target_y[n_iteration])

        self.target_x[n_iteration].append(config)
        self.target_y[n_iteration].append(perf)

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)
            # Update history container.
            self.history_container.add(config, perf)

            # Update weight
            if self.update_enable and len(self.incumbent_configs) >= 8:  # todo: replace 8 by full observation num
                self.weight_update_id += 1
                self.update_weight()

        # Refit the ensemble surrogate model. todo: no median
        configs_train = self.target_x[n_iteration]
        results_train = self.target_y[n_iteration]
        results_train = np.array(std_normalization(results_train), dtype=np.float64)
        self.surrogate.train(convert_configurations_to_array(configs_train), results_train, r=n_iteration)

    def choose_next(self):
        """
        sample a config according to MFES. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        # sample config
        excluded_configs = self.bracket[next_rung_id]['configs']
        if len(self.target_y[self.iterate_r[-1]]) < 3:      # todo: caution
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # Like BOHB, sample a fixed percentage of random configurations.
            self.random_check_idx += 1
            if self.random_configuration_chooser.check(self.random_check_idx):
                next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
            else:
                acq_configs = self.get_bo_candidates()
                for config in acq_configs:
                    if config not in self.bracket[next_rung_id]['configs']:
                        next_config = config
                        break
                if next_config is None:
                    self.logger.warning('Cannot get a non duplicate configuration from bo candidates. '
                                        'Sample a random one.')
                    next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_bo_candidates(self):
        # std_incumbent_value = np.min(std_normalization(self.target_y[self.iterate_r[-1]]))
        # # Update surrogate model in acquisition function.
        # self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
        #                                  num_data=len(self.incumbent_configs))
        #
        # challengers = self.acq_optimizer.maximize(
        #     runhistory=self.history_container,
        #     num_points=5000,
        # )
        # return challengers.challengers
        best_index = np.argmin(self.incumbent_perfs)
        best_config = self.incumbent_configs[best_index]
        std_incumbent_value = np.min(std_normalization(self.incumbent_perfs))
        # Update surrogate model in acquisition function.
        self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                         num_data=len(self.incumbent_configs))
        candidates = self.acq_optimizer.maximize(best_config=best_config, batch_size=5000)
        return candidates       # todo

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def update_weight(self):
        return  # debug

    def get_weights(self):
        return self.hist_weights
