import os
import time
import numpy as np

from mq_hb.async_mq_sh import async_mqSuccessiveHalving
from mq_hb.utils import RUNNING, COMPLETED, PROMOTED
from mq_hb.utils import sample_configuration
from mq_hb.utils import minmax_normalization, std_normalization

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.utils.config_space.util import convert_configurations_to_array

from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.utils.constants import MAXINT
from mq_hb.utils import sample_configurations


class RandomSampling(object):

    def __init__(self, objective_function, config_space, n_samples=5000, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        n_samples: int
            Number of candidates that are samples
        """
        self.config_space = config_space
        self.objective_func = objective_function
        if rng is None:
            self.rng = np.random.RandomState(1357)
        else:
            self.rng = rng
        self.n_samples = n_samples

    def maximize(self, best_config, batch_size=1):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------
        batch_size: number of maximizer returned.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        incs_configs = list(get_one_exchange_neighbourhood(best_config, seed=self.rng.randint(MAXINT)))

        # Sample random points uniformly over the whole space
        rand_configs = sample_configurations(self.config_space, max(self.n_samples, batch_size) - len(incs_configs))

        configs_list = incs_configs + rand_configs

        y = self.objective_func(configs_list)
        if batch_size == 1:
            return [configs_list[np.argmax(y)]]

        candidates = [configs_list[int(i)] for i in np.argsort(y)[-batch_size:][::-1]]  # fix: [::-1]
        return candidates


class async_mqBOSH(async_mqSuccessiveHalving):
    """
    The implementation of Asynchronous BOSH (combine ASHA and BOHB)
    no Hyperband
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 rand_prob=0.3,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqAsyncBOSH',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc'):
        super().__init__(objective_func, config_space, R, eta=eta,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        types, bounds = get_types(config_space)
        self.surrogate = RandomForestWithInstances(types=types, bounds=bounds)
        self.acquisition_function = EI(model=self.surrogate)
        self.acq_optimizer = RandomSampling(self.acquisition_function, config_space,
                                            n_samples=max(5000, 50 * len(bounds)))
        self.rng = np.random.RandomState(self.seed)

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

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)
            # train BO surrogate
            train_perfs = np.array(std_normalization(self.incumbent_perfs), dtype=np.float64)
            self.surrogate.train(convert_configurations_to_array(self.incumbent_configs), train_perfs)

    def choose_next(self):
        """
        sample a config according to BO.
        """
        next_config = None
        next_n_iteration = self.bracket[0]['n_iteration']
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        # sample config
        excluded_configs = self.bracket[next_rung_id]['configs']

        if len(self.incumbent_configs) < self.bo_init_num or self.rng.random() < self.rand_prob:
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # BO
            start_time = time.time()
            best_index = np.argmin(self.incumbent_perfs)
            best_config = self.incumbent_configs[best_index]
            std_incumbent_value = np.min(std_normalization(self.incumbent_perfs))
            # Update surrogate model in acquisition function.
            self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                             num_data=len(self.incumbent_configs))
            candidates = self.acq_optimizer.maximize(best_config=best_config, batch_size=5000)
            time1 = time.time()
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
                self.logger.info('BO opt cost %.2fs. check duplication cost %.2fs. len of incumbent_configs: %d.'
                                 % (time1-start_time, time2-time1, len(self.incumbent_configs)))

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf
