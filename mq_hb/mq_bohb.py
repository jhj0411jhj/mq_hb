import numpy as np

from mq_hb.utils import sample_configurations, expand_configurations
from mq_hb.mq_hb import mqHyperband

try:
    from litebo.config_space import ConfigurationSpace
except ImportError as e:
    from litebo.utils.config_space import ConfigurationSpace
from litebo.core.sync_batch_advisor import SyncBatchAdvisor, SUCCESS


class mqBOHB(mqHyperband):
    """ The implementation of BOHB.
        The paper can be found in https://arxiv.org/abs/1807.01774 .
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 num_iter=10000,
                 rand_prob=0.3,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqBOHB',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 ip='',
                 port=13579, ):
        super().__init__(objective_func, config_space, R, eta=eta, num_iter=num_iter,
                         random_state=random_state, method_id=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         ip=ip, port=port)

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        self.rng = np.random.RandomState(random_state)
        self.config_advisor = SyncBatchAdvisor(self.config_space,
                                               batch_size=None,
                                               batch_strategy='median_imputation',
                                               surrogate_type='prf',
                                               initial_trials=self.bo_init_num,
                                               init_strategy='random_explore_first',
                                               optimization_strategy='bo',
                                               task_id=None,
                                               output_dir=self.log_directory,
                                               rng=self.rng)
        self.config_advisor.optimizer.rand_prob = 0.0

    def choose_next(self, num_config):
        # Sample n configurations according to BOHB strategy.
        self.logger.info('Sample %d configs in choose_next. rand_prob is %f.' % (num_config, self.rand_prob))

        # get bo configs
        # update batchsize each round. todo: random ratio is fixed?
        self.config_advisor.batch_size = num_config - int(num_config * self.rand_prob)
        bo_configs = self.config_advisor.get_suggestions()
        bo_configs = bo_configs[:num_config]    # may exceed num_config in initial random sampling
        self.logger.info('len bo configs = %d.' % len(bo_configs))

        # sample random configs
        configs = expand_configurations(bo_configs, self.config_space, num_config)
        self.logger.info('len total configs = %d.' % len(configs))
        assert len(configs) == num_config
        return configs

    def update_incumbent_before_reduce(self, T, val_losses, n_iterations):
        if int(n_iterations) < self.R:
            return
        self.incumbent_configs.extend(T)
        self.incumbent_perfs.extend(val_losses)
        # update config advisor
        for config, perf in zip(T, val_losses):
            observation = (config, perf, SUCCESS)
            self.config_advisor.update_observation(observation)
            self.logger.info('update observation: %s' % str(observation))
        self.logger.info('%d observations updated. %d incumbent configs total.' % (len(T), len(self.incumbent_configs)))

    def update_incumbent_after_reduce(self, T, incumbent_loss):
        return
