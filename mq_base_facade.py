import time
import os
from litebo.utils.logging_utils import get_logger, setup_logger
from litebo.core.message_queue.master_messager import MasterMessager


class mqBaseFacade(object):
    def __init__(self, objective_func,
                 restart_needed=False,
                 need_lc=False,
                 method_name=None,
                 log_directory='logs',
                 max_queue_len=100,
                 ip='',
                 port=13579,):
        self.log_directory = log_directory
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        self.logger = self._get_logger("%s-%s" % (__class__.__name__, method_name))

        self.objective_func = objective_func
        self.trial_statistics = []
        self.recorder = []

        self.global_start_time = time.time()
        self.runtime_limit = None
        self._history = {"time_elapsed": [], "performance": [], "best_trial_id": [], "configuration": []}
        self.global_incumbent = 1e10
        self.global_incumbent_configuration = None
        self.global_trial_counter = 0
        self.restart_needed = restart_needed
        self.record_lc = need_lc
        self.method_name = method_name
        # evaluation metrics
        self.stage_id = 1
        self.stage_history = {'stage_id': [], 'performance': []}
        self.grid_search_perf = []

        if self.method_name is None:
            raise ValueError('Method name must be specified! NOT NONE.')

        self.time_limit_per_trial = 60     # todo caution

        max_queue_len = max(100, max_queue_len)
        self.master_messager = MasterMessager(ip, port, max_queue_len, max_queue_len)

    def set_restart(self):
        self.restart_needed = True

    def set_method_name(self, name):
        self.method_name = name

    def add_stage_history(self, stage_id, performance):
        self.stage_history['stage_id'].append(stage_id)
        self.stage_history['performance'].append(performance)

    def add_history(self, time_elapsed, performance, trial_id, config):
        self._history['time_elapsed'].append(time_elapsed)
        self._history['performance'].append(performance)
        self._history['best_trial_id'].append(trial_id)
        self._history['configuration'].append(config)

    def run_in_parallel(self, configurations, n_iteration, extra_info=None):
        n_configuration = len(configurations)
        performance_result = []
        early_stops = []

        # TODO: need systematic tests.
        # check configurations, whether it exists the same configs
        count_dict = dict()
        for i, config in enumerate(configurations):
            if config not in count_dict:
                count_dict[config] = 0
            count_dict[config] += 1

        # incorporate ref info.
        conf_list = []
        for index, config in enumerate(configurations):
            conf_dict = config.get_dictionary().copy()
            if count_dict[config] > 1:
                conf_dict['uid'] = count_dict[config]
                count_dict[config] -= 1

            if extra_info is not None:
                conf_dict['reference'] = extra_info[index]
            conf_dict['need_lc'] = self.record_lc
            conf_dict['method_name'] = self.method_name
            conf_list.append(conf_dict)

        # Add batch configs to masterQueue.
        for config in conf_list:
            msg = [config, self.time_limit_per_trial, n_iteration, self.global_trial_counter]
            self.master_messager.send_message(msg)
            self.global_trial_counter += 1
        self.logger.info('Master: %d configs sent.' % (len(conf_list)))
        # Get batch results from workerQueue.
        result_num = 0
        result_needed = len(conf_list)
        while True:
            observation = self.master_messager.receive_message()
            if observation is None:
                # Wait for workers.
                # self.logger.info("Master: wait for worker results. sleep 1s.")
                time.sleep(1)
                continue
            # Report result.
            result_num += 1
            self.trial_statistics.append(observation)   # return_info, time_taken, trail_id, config
            self.logger.info('Master: Get the [%d] result, observation is %s.' % (result_num, str(observation)))
            if result_num == result_needed:
                break

        # get the evaluation statistics
        for trial in self.trial_statistics:
            return_info, time_taken, trail_id, config = trial

            performance = return_info['loss']
            if performance < self.global_incumbent:
                self.global_incumbent = performance
                self.global_incumbent_configuration = config

            self.add_history(time.time() - self.global_start_time, self.global_incumbent, trail_id,
                             self.global_incumbent_configuration)
            # TODO: old version => performance_result.append(performance)
            performance_result.append(return_info)
            early_stops.append(return_info.get('early_stop', False))
            self.recorder.append({'trial_id': trail_id, 'time_consumed': time_taken,
                                  'configuration': config, 'n_iteration': n_iteration})

        self.trial_statistics.clear()

        if self.runtime_limit is not None and time.time() - self.global_start_time > self.runtime_limit:
            raise ValueError('Runtime budget meets!')
        return performance_result, early_stops

    def _get_logger(self, name):
        logger_name = 'mfes_%s' % name
        setup_logger(os.path.join(self.log_directory, '%s.log' % str(logger_name)), None)
        return get_logger(logger_name)
