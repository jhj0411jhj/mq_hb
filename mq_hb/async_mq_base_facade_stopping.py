import time
import os
import traceback
import numpy as np
import pickle as pkl
from openbox.utils.logging_utils import get_logger, setup_logger
from openbox.core.message_queue.master_messager import MasterMessager
from mq_hb.message_queue.sender_messager import SenderMessager

PLOT = False
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    PLOT = True
except Exception as e:
    pass


class async_mqBaseFacade_stopping(object):
    def __init__(self, objective_func,
                 restart_needed=False,
                 need_lc=False,
                 method_name='default_method_name',
                 log_directory='logs',
                 data_directory='data',
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 max_queue_len=1000,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 sleep_time=0.1,):
        self.log_directory = log_directory
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        self.data_directory = data_directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        self.logger = self._get_logger(method_name)

        self.objective_func = objective_func
        self.trial_statistics = list()
        self.recorder = list()

        self.global_start_time = time.time()
        self.runtime_limit = None
        self._history = {"time_elapsed": list(), "performance": list(),
                         "best_trial_id": list(), "configuration": list()}
        self.global_incumbent = 1e10
        self.global_incumbent_configuration = None
        self.global_trial_counter = 0
        self.restart_needed = restart_needed
        self.record_lc = need_lc
        self.method_name = method_name
        # evaluation metrics
        self.stage_id = 1
        self.stage_history = {'stage_id': list(), 'performance': list()}
        self.grid_search_perf = list()

        self.save_intermediate_record = False
        self.save_intermediate_record_id = 0
        self.save_intermediate_record_path = None

        if self.method_name is None:
            raise ValueError('Method name must be specified! NOT NONE.')

        self.time_limit_per_trial = time_limit_per_trial
        self.runtime_limit = runtime_limit
        assert self.runtime_limit is not None

        max_queue_len = max(1000, max_queue_len)
        self.master_messager = MasterMessager(ip, port, authkey, max_queue_len, max_queue_len)
        self.sleep_time = sleep_time

        self.sender_dict = dict()

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

    def run(self):
        try:
            while True:
                if self.runtime_limit is not None and time.time() - self.global_start_time > self.runtime_limit:
                    self.logger.info('RUNTIME BUDGET is RUNNING OUT.')
                    return

                # Get observation from worker
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    time.sleep(self.sleep_time)
                    continue

                return_info, time_taken, trial_id, config, worker_info = observation
                if config is None:
                    # worker free, send new job
                    self.logger.info('Receive init message.')
                    t = time.time()
                    config, n_iteration, extra_conf = self.get_job()
                    self.logger.info('get_job() cost %.2fs.' % (time.time() - t,))
                    msg = [config, extra_conf, self.time_limit_per_trial, n_iteration, self.global_trial_counter]
                    self.master_messager.send_message(msg)
                    self.global_trial_counter += 1
                    self.logger.info('Master send job: %s.' % (msg,))
                else:
                    # decide stopping
                    global_time = time.time() - self.global_start_time
                    self.logger.info('Master get observation: %s. Global time=%.2fs.' % (str(observation), global_time))
                    n_iteration = return_info['n_iteration']
                    perf = return_info['loss']
                    t = time.time()
                    next_n_iteration = self.decide_stopping(config, perf, n_iteration)
                    self.logger.info('decide_stopping() cost %.2fs.' % (time.time() - t,))
                    self.recorder.append({'trial_id': trial_id, 'time_consumed': time_taken,
                                          'configuration': config, 'n_iteration': n_iteration,
                                          'return_info': return_info, 'global_time': global_time})
                    if (not hasattr(self, 'R')) or n_iteration == self.R:
                        self.save_intermediate_statistics()

                    # send decision
                    try:
                        worker_ip = worker_info['ip']
                        worker_port = worker_info['port']
                        worker_authkey = worker_info['authkey']

                        if (worker_ip, worker_port, worker_authkey) in self.sender_dict.keys():
                            sender_messager = self.sender_dict[(worker_ip, worker_port, worker_authkey)]
                        else:
                            sender_messager = SenderMessager(worker_ip, worker_port, worker_authkey)
                            self.sender_dict[(worker_ip, worker_port, worker_authkey)] = sender_messager
                        msg = next_n_iteration
                        sender_messager.send_message(msg)
                    except Exception:
                        self.logger.error('Send decision error: %s' % traceback.format_exc())
                    self.logger.info('Num of senders: %d' % len(self.sender_dict))

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.logger.error(traceback.format_exc())

    def get_job(self):
        raise NotImplementedError

    def decide_stopping(self, config, perf, n_iteration):
        raise NotImplementedError

    def set_save_intermediate_record(self, dir_path, file_name):
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            pass
        self.save_intermediate_record = True
        if file_name.endswith('.pkl'):
            file_name = file_name[:-4]
        self.save_intermediate_record_path = os.path.join(dir_path, file_name)
        self.logger.info('set save_intermediate_record to True. path: %s.' % (self.save_intermediate_record_path,))

    def save_intermediate_statistics(self):
        if self.save_intermediate_record:
            self.save_intermediate_record_id += 1
            path = '%s_%d.pkl' % (self.save_intermediate_record_path, self.save_intermediate_record_id)
            with open(path, 'wb') as f:
                pkl.dump(self.recorder, f)
            global_time = time.time() - self.global_start_time
            self.logger.info('Intermediate record %s saved! global_time=%.2fs.' % (path, global_time))

    def _get_logger(self, name):
        logger_name = name
        setup_logger(os.path.join(self.log_directory, '%s.log' % str(logger_name)), None)
        return get_logger(self.__class__.__name__)
