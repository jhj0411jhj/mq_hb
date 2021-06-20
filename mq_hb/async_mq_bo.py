import numpy as np
from mq_hb.async_mq_base_facade import async_mqBaseFacade
from mq_hb.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace
from openbox.core.async_batch_advisor import AsyncBatchAdvisor, SUCCESS
from openbox.core.base import Observation


class async_mqBO(async_mqBaseFacade):
    """
    The implementation of Asynchronous Parallel Bayesian Optimization (using OpenBox)
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqAsyncBO',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 **kwargs):
        max_queue_len = 1000   # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)
        self.R = R

        self.incumbent_configs = list()
        self.incumbent_perfs = list()

        self.logger.info('Unused kwargs: %s' % kwargs)

        self.bo_init_num = bo_init_num
        task_info = {'num_constraints': 0, 'num_objs': 1}
        # using median_imputation batch_strategy implemented in OpenBox to generate BO suggestions
        self.config_advisor = AsyncBatchAdvisor(config_space, task_info,
                                                batch_size=None,
                                                batch_strategy='median_imputation',
                                                initial_trials=self.bo_init_num,
                                                init_strategy='random',
                                                optimization_strategy='bo',
                                                surrogate_type='prf',
                                                acq_type='ei',
                                                acq_optimizer_type='local_random',
                                                task_id=self.method_name,
                                                output_dir=self.log_directory,
                                                random_state=random_state,
                                                )

    def get_job(self):
        """
        sample config
        """
        next_config = self.config_advisor.get_suggestion()
        next_n_iteration = self.R
        next_extra_conf = dict(initial_run=True)

        return next_config, next_n_iteration, next_extra_conf

    def update_observation(self, config, perf, n_iteration):
        assert int(n_iteration) == self.R
        self.incumbent_configs.append(config)
        self.incumbent_perfs.append(perf)

        # update bo advisor
        objs = [perf]
        # config, trial_state, constraints, objs, elapsed_time
        observation = Observation(config, SUCCESS, None, objs, None)
        self.config_advisor.update_observation(observation)
        self.logger.info('update BO observation: config=%s, perf=%f' % (str(config), perf))

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
