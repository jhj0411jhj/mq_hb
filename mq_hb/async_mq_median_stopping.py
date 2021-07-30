import time
import numpy as np
from math import log, ceil
from typing import List
from mq_hb.async_mq_base_facade_stopping import async_mqBaseFacade_stopping
from mq_hb.utils import RUNNING, COMPLETED, STOPPED
from mq_hb.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.constants import MAXINT


class Job(object):
    def __init__(self, config, extra_conf=None):
        self.config = config
        self.extra_conf = extra_conf if extra_conf is not None else dict()
        self.job_status = RUNNING
        self.perfs = list()
        self.best_perf = MAXINT

    def add_perf(self, perf, n_iteration):
        assert n_iteration == len(self.perfs) + 1   # n_iteration starts from 1
        self.perfs.append(perf)
        self.best_perf = min(self.best_perf, perf)

    def __str__(self):
        return 'Job(config:%s, status:%s, best_perf:%s, n_perfs:%d)'\
               % (self.config, self.job_status, self.best_perf, len(self.perfs))

    def __repr__(self):
        return self.__str__()


class async_mqMedianStopping(async_mqBaseFacade_stopping):
    """
    The implementation of median stopping in Google Vizier
    + Report at each n_iteration in [1, R]
    + Decide stopping at n_iteration in stop_iterations
    + Random Sampling
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 stop_iterations: List = None,
                 window_size: int = None,
                 random_state=1,
                 method_id='mqMedianStopping',
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

        self.R = R      # Maximum iterations per configuration
        self.stop_iterations = stop_iterations  # n_iterations to decide stopping
        if self.stop_iterations is None:
            self.stop_iterations = list(range(1, 1 + R))
        self.window_size = window_size  # size for perf averaging
        assert self.window_size is None or self.window_size > 0
        self.logger.info('R: %d. window_size: %d. stop_iterations: %s.'
                         % (self.R, self.window_size, self.stop_iterations))

        self.incumbent_configs = list()
        self.incumbent_perfs = list()

        self.all_configs = set()
        self.job_dict = dict()
        self.complete_perfs = list()
        print('unused kwargs: %s.', kwargs)

    def get_median_value(self, n_iteration):
        if len(self.complete_perfs) == 0:
            return None

        start_n_iteration = 0 if self.window_size is None else max(0, n_iteration - self.window_size)
        complete_perfs = np.array(self.complete_perfs, dtype=np.float64)
        running_average = np.average(complete_perfs[:, start_n_iteration:n_iteration], axis=1)
        median_value = np.median(running_average)
        return median_value

    def get_job(self):
        """
        sample a new config
        """
        next_config, next_n_iteration, next_extra_conf = self.choose_next()
        self.logger.info('Sample a new config: %s. next_n_iteration: %d.' % (next_config, next_n_iteration))

        assert next_config not in self.all_configs
        self.all_configs.add(next_config)
        self.job_dict[next_config] = Job(next_config, next_extra_conf)

        return next_config, next_n_iteration, next_extra_conf

    def decide_stopping(self, config, perf, n_iteration):
        """
        update observation and decide stopping
        return -1 if stopping
        """
        job = self.job_dict[config]
        job.add_perf(perf, n_iteration)

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)

            self.logger.info('Complete job: %s.' % (job, ))
            job.job_status = COMPLETED
            assert len(job.perfs) == self.R
            self.complete_perfs.append(job.perfs)
            return -1

        next_n_iteration = n_iteration + 1
        if n_iteration not in self.stop_iterations:
            self.logger.info('Continue job: %s. next_n_iteration: %d.' % (job, next_n_iteration))
            return next_n_iteration

        median_value = self.get_median_value(n_iteration)
        if median_value is None or job.best_perf <= median_value:
            self.logger.info('Continue job: %s. median_value: %s. next_n_iteration: %d.'
                             % (job, str(median_value), next_n_iteration))
            return next_n_iteration
        else:
            self.logger.info('Stop job: %s. median_value: %s. next_n_iteration: %d.'
                             % (job, str(median_value), next_n_iteration))
            job.job_status = STOPPED
            return -1

    def choose_next(self):
        """
        sample a random config and give the least iterations
        """
        next_config = sample_configuration(self.config_space, excluded_configs=self.all_configs)
        next_n_iteration = 1
        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
