import os
import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from typing import List

from mq_hb.async_mq_median_stopping import async_mqMedianStopping
from mq_hb.utils import sample_configuration
from mq_hb.utils import minmax_normalization, std_normalization
from mq_hb.surrogate.rf_ensemble import RandomForestEnsemble2
from mq_hb.surrogate.gp_ensemble import GaussianProcessEnsemble2
from mq_hb.acq_maximizer.ei_optimization import RandomSampling

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer


class async_mqMFES_MedianStopping(async_mqMedianStopping):
    """
    The implementation of Asynchronous MFES
    + Build models at model_iterations
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
                 rand_prob=0.3,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm',
                 fusion_method='idp',
                 power_num=3,
                 non_decreasing_weight=False,
                 increasing_weight=True,
                 surrogate_type='prf',  # 'prf', 'gp'
                 acq_optimizer='local_random',  # 'local_random', 'random'
                 median_imputation=None,  # None, 'top', 'corresponding', 'all'
                 test_random=False,
                 test_bohb=False,
                 random_state=1,
                 method_id='mqAsyncMFES_median',
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

        # test version
        self.test_random = test_random
        self.test_bohb = test_bohb

        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Parameter for weight method `rank_loss_p_norm`.
        self.power_num = power_num
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        self.init_weight = init_weight
        if self.init_weight is None:
            n_remain = len(self.iterate_r) - 1
            if self.test_bohb:
                self.init_weight = [0.] * n_remain + [1.]
            elif n_remain == 0:
                self.init_weight = [1.]
            else:
                self.init_weight = [1. / n_remain] * n_remain + [0.]
        assert len(self.init_weight) == len(self.iterate_r)
        self.logger.info("Initialize weight to %s" % self.init_weight)
        types, bounds = get_types(config_space)

        self.rng = np.random.RandomState(seed=self.seed)

        self.surrogate_type = surrogate_type
        if self.surrogate_type == 'prf':
            self.surrogate = RandomForestEnsemble2(types, bounds, self.iterate_r,
                                                   self.init_weight, self.fusion_method)
        elif self.surrogate_type == 'gp':
            self.surrogate = GaussianProcessEnsemble2(config_space, types, bounds, self.iterate_r,
                                                      self.init_weight, self.fusion_method, self.rng)
        else:
            raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
        self.acquisition_function = EI(model=self.surrogate)

        self.iterate_id = 0
        self.hist_weights = list()
        self.hist_weights_unadjusted = list()

        # Saving evaluation statistics
        self.target_x = dict()
        self.target_y = dict()
        for index, r in enumerate(self.iterate_r):
            self.target_x[r] = list()
            self.target_y[r] = list()

        # BO optimizer settings.
        self.history_container = HistoryContainer(task_id=self.method_name)
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.acq_optimizer_type = acq_optimizer
        if self.acq_optimizer_type == 'local_random':
            self.acq_optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=self.acquisition_function,
                config_space=self.config_space,
                rng=self.rng,
                max_steps=self.sls_max_steps,
                n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
                n_sls_iterations=self.n_sls_iterations,
                rand_prob=0.0,
            )
        elif self.acq_optimizer_type == 'random':
            self.acq_optimizer = RandomSampling(self.acquisition_function, config_space,
                                                n_samples=max(5000, 50 * len(bounds)))
        else:
            raise ValueError
        self.random_configuration_chooser = ChooserProb(prob=rand_prob, rng=self.rng)
        self.random_check_idx = 0

        self.non_decreasing_weight = non_decreasing_weight
        self.increasing_weight = increasing_weight
        assert not (self.non_decreasing_weight and self.increasing_weight)

        # median imputation
        self.median_imputation = median_imputation
        self.configs_running_dict = dict()
        self.all_configs_running = set()
        assert self.median_imputation in [None, 'top', 'corresponding', 'all']

    def get_job(self):
        next_config, next_n_iteration, next_extra_conf = super().get_job()
        # for median imputation
        if self.median_imputation is not None:
            raise NotImplementedError

        return next_config, next_n_iteration, next_extra_conf

    def train_surrogate(self, surrogate, n_iteration: int, median_imputation: bool, all_impute=False):
        if median_imputation:
            raise NotImplementedError
        else:
            configs_train = self.target_x[n_iteration]
            results_train = self.target_y[n_iteration]
        results_train = np.array(std_normalization(results_train), dtype=np.float64)
        surrogate.train(convert_configurations_to_array(configs_train), results_train, r=n_iteration)

    def decide_stopping(self, config, perf, n_iteration):
        decision = super().decide_stopping(config, perf, n_iteration)

        if n_iteration in self.iterate_r:
            self.target_x[n_iteration].append(config)
            self.target_y[n_iteration].append(perf)

        if self.median_imputation is not None:
            raise NotImplementedError

        # Refit the ensemble surrogate model.
        if n_iteration in self.iterate_r:
            start_time = time.time()
            if self.median_imputation is None:
                self.train_surrogate(self.surrogate, n_iteration, median_imputation=False)
            else:
                raise NotImplementedError
            self.logger.info('update_observation training surrogate cost %.2fs.' % (time.time() - start_time))
        else:
            self.logger.info('n_iteration not in model_iterations. Skip training.')

        if n_iteration == self.R:
            # self.incumbent_configs.append(config)
            # self.incumbent_perfs.append(perf)
            # Update history container.
            self.history_container.add(config, perf)

            # Update weight
            if self.update_enable and len(self.incumbent_configs) >= 8:  # todo: replace 8 by full observation num
                self.weight_update_id += 1
                self.update_weight()

        return decision

    def choose_next(self):
        """
        sample a config according to MFES. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()

        # sample config
        excluded_configs = self.all_configs
        if any([len(y_list) == 0 for y_list in self.target_y.values()]) or self.test_random:
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # Like BOHB, sample a fixed percentage of random configurations.
            self.random_check_idx += 1
            if self.random_configuration_chooser.check(self.random_check_idx):
                next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
            else:
                acq_configs = self.get_bo_candidates()
                for config in acq_configs:
                    if config not in excluded_configs:
                        next_config = config
                        break
                if next_config is None:
                    self.logger.warning('Cannot get a non duplicate configuration from bo candidates. '
                                        'Sample a random one.')
                    next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_next_n_iteration(self):
        return 1

    def get_bo_candidates(self):
        if self.acq_optimizer_type == 'local_random':
            std_incumbent_value = np.min(std_normalization(self.target_y[self.iterate_r[-1]]))
            # Update surrogate model in acquisition function.
            self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                             num_data=len(self.incumbent_configs))

            challengers = self.acq_optimizer.maximize(
                runhistory=self.history_container,
                num_points=5000,
            )
            return challengers.challengers
        elif self.acq_optimizer_type == 'random':
            best_index = np.argmin(self.incumbent_perfs)
            best_config = self.incumbent_configs[best_index]
            std_incumbent_value = np.min(std_normalization(self.incumbent_perfs))
            # Update surrogate model in acquisition function.
            self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                             num_data=len(self.incumbent_configs))
            candidates = self.acq_optimizer.maximize(best_config=best_config, batch_size=5000)
            return candidates
        else:
            raise ValueError

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
        start_time = time.time()

        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        if len(incumbent_configs) < 3:
            return
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        r_list = self.surrogate.surrogate_r
        K = len(r_list)

        old_weights = list()
        for i, r in enumerate(r_list):
            _weight = self.surrogate.surrogate_weight[r]
            old_weights.append(_weight)

        if len(test_y) >= 3:
            # refit surrogate model without median imputation
            if self.median_imputation is None:
                test_surrogate = self.surrogate
            else:
                types, bounds = get_types(self.config_space)
                if self.surrogate_type == 'prf':
                    test_surrogate = RandomForestEnsemble2(types, bounds, self.iterate_r,
                                                           old_weights, self.fusion_method)
                elif self.surrogate_type == 'gp':
                    test_surrogate = GaussianProcessEnsemble2(self.config_space, types, bounds, self.iterate_r,
                                                              old_weights, self.fusion_method, self.rng)
                else:
                    raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
                for r in self.iterate_r:
                    self.train_surrogate(test_surrogate, r, median_imputation=False)

            # Get previous weights
            if self.weight_method == 'rank_loss_p_norm':
                preserving_order_p = list()
                preserving_order_nums = list()
                for i, r in enumerate(r_list):
                    fold_num = 5
                    if i != K - 1:
                        mean, var = test_surrogate.surrogate_container[r].predict(test_x)
                        tmp_y = np.reshape(mean, -1)
                        preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                        preserving_order_p.append(preorder_num / pair_num)
                        preserving_order_nums.append(preorder_num)
                    else:
                        if len(test_y) < 2 * fold_num:
                            preserving_order_p.append(0)
                        else:
                            # 5-fold cross validation.
                            kfold = KFold(n_splits=fold_num)
                            cv_pred = np.array([0] * len(test_y))
                            for train_idx, valid_idx in kfold.split(test_x):
                                train_configs, train_y = test_x[train_idx], test_y[train_idx]
                                valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                                types, bounds = get_types(self.config_space)
                                if self.surrogate_type == 'prf':
                                    _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                                elif self.surrogate_type == 'gp':
                                    _surrogate = create_gp_model('gp', self.config_space, types, bounds, self.rng)
                                else:
                                    raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
                                _surrogate.train(train_configs, train_y)
                                pred, _ = _surrogate.predict(valid_configs)
                                cv_pred[valid_idx] = pred.reshape(-1)
                            preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, test_y)
                            preserving_order_p.append(preorder_num / pair_num)
                            preserving_order_nums.append(preorder_num)

                self.logger.info('update weight preserving_order_p: %s' % preserving_order_p)
                trans_order_weight = np.array(preserving_order_p)
                power_sum = np.sum(np.power(trans_order_weight, self.power_num))
                new_weights = np.power(trans_order_weight, self.power_num) / power_sum

            elif self.weight_method == 'rank_loss_prob':
                t1 = time.time()
                # For basic surrogate i=1:K-1.
                mean_list, var_list = list(), list()
                prob_list = list()
                for i, r in enumerate(r_list[:-1]):
                    mean, var = test_surrogate.surrogate_container[r].predict(test_x)
                    mean_list.append(np.reshape(mean, -1))
                    var_list.append(np.reshape(var, -1))

                    tmp_y = np.reshape(mean, -1)
                    preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                    prob_list.append(preorder_num / pair_num)
                self.logger.info('update weight preserving_order prob_list: %s' % prob_list)

                t2 = time.time()

                sample_num = 100
                min_probability_array = [0] * K
                for _ in range(sample_num):
                    order_preseving_nums = list()

                    # For basic surrogate i=1:K-1.
                    for idx in range(K - 1):
                        sampled_y = self.rng.normal(mean_list[idx], var_list[idx])
                        _num, _ = self.calculate_preserving_order_num(sampled_y, test_y)
                        order_preseving_nums.append(_num)

                    fold_num = 5
                    # For basic surrogate i=K. cv
                    if len(test_y) < 2 * fold_num or self.increasing_weight:
                        order_preseving_nums.append(0)
                    else:
                        # 5-fold cross validation.
                        kfold = KFold(n_splits=fold_num)
                        cv_pred = np.array([0] * len(test_y))
                        for train_idx, valid_idx in kfold.split(test_x):    # todo: reduce cost!!!
                            train_configs, train_y = test_x[train_idx], test_y[train_idx]
                            valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                            types, bounds = get_types(self.config_space)
                            if self.surrogate_type == 'prf':
                                _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                            elif self.surrogate_type == 'gp':
                                _surrogate = create_gp_model('gp', self.config_space, types, bounds, self.rng)
                            else:
                                raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
                            _surrogate.train(train_configs, train_y)
                            _pred, _var = _surrogate.predict(valid_configs)
                            sampled_pred = self.rng.normal(_pred.reshape(-1), _var.reshape(-1))
                            cv_pred[valid_idx] = sampled_pred
                        _num, _ = self.calculate_preserving_order_num(cv_pred, test_y)
                        order_preseving_nums.append(_num)
                    max_id = np.argmax(order_preseving_nums)
                    min_probability_array[max_id] += 1
                new_weights = np.array(min_probability_array) / sample_num
                t3 = time.time()
                if t3 - t1 > 1:
                    self.logger.info('update weight (rank loss prob) cost time: %.2fs, %.2fs'
                                     % (t2 - t1, t3 - t2))
            else:
                raise ValueError('Invalid weight method: %s!' % self.weight_method)
        else:
            new_weights = np.array(old_weights)

        # non decreasing full observation weight
        old_weights = np.asarray(old_weights)
        new_weights = np.asarray(new_weights)
        self.hist_weights_unadjusted.append(new_weights)
        if self.non_decreasing_weight:
            old_last_weight = old_weights[-1]
            new_last_weight = new_weights[-1]
            if new_last_weight < old_last_weight:
                old_remain_weight = 1.0 - old_last_weight
                new_remain_weight = 1.0 - new_last_weight
                if new_remain_weight <= 1e-8:
                    adjusted_new_weights = np.array([0.] * (K-1) + [1.], dtype=np.float64)
                else:
                    adjusted_new_weights = np.append(new_weights[:-1] / new_remain_weight * old_remain_weight,
                                                     old_last_weight)
                self.logger.info('[%s] %d-th. non_decreasing_weight: old_weights=%s, new_weights=%s, '
                                 'adjusted_new_weights=%s.' % (self.weight_method, self.weight_changed_cnt,
                                                               old_weights, new_weights, adjusted_new_weights))
                new_weights = adjusted_new_weights
        elif self.increasing_weight and len(test_y) >= 10:
            s = 10
            k = 0.025
            a = 0.5
            new_last_weight = a / (a + np.e ** (-(len(test_y) - s) * k))
            new_remain_weight = 1.0 - new_last_weight
            remain_weight = 1.0 - new_weights[-1]
            if remain_weight <= 1e-8:
                adjusted_new_weights = np.array([0.] * (K-1) + [1.], dtype=np.float64)
            else:
                adjusted_new_weights = np.append(new_weights[:-1] / remain_weight * new_remain_weight,
                                                 new_last_weight)
            self.logger.info('[%s] %d-th. increasing_weight: new_weights=%s, adjusted_new_weights=%s.'
                             % (self.weight_method, self.weight_changed_cnt, new_weights, adjusted_new_weights))
            new_weights = adjusted_new_weights

        self.logger.info('[%s] %d-th Updating weights: %s' % (
            self.weight_method, self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        if not self.test_bohb:
            for i, r in enumerate(r_list):
                self.surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        self.hist_weights.append(new_weights)
        dir_path = os.path.join(self.data_directory, 'saved_weights')
        file_name = 'mfes_weights_%s.npy' % (self.method_name,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(os.path.join(dir_path, file_name), np.asarray(self.hist_weights))
        self.logger.info('update_weight() cost %.2fs. new weights are saved to %s'
                         % (time.time() - start_time, os.path.join(dir_path, file_name)))

    def get_weights(self):
        return self.hist_weights
