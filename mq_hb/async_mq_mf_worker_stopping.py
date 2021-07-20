import sys
import time
import traceback
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.core.message_queue.worker_messager import WorkerMessager
from mq_hb.utils import get_host_ip, StoppingException
from mq_hb.message_queue.reporter import Reporter


def no_time_limit_func(objective_function, time_limit_per_trial, args, kwargs):
    ret = objective_function(*args, **kwargs)
    return False, ret


class async_mqmfWorker_stopping(object):
    """
    async message queue worker for multi-fidelity optimization
    stopping variant
    """
    def __init__(self, objective_function,
                 ip="127.0.0.1", port=13579, authkey=b'abc',
                 self_ip=None, self_port=13531, self_authkey=b'abc',
                 sleep_time=0.1,
                 no_time_limit=True,    # todo: time_limit
                 logger=None):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port, authkey=authkey)
        self.sleep_time = sleep_time

        if no_time_limit:
            self.time_limit = no_time_limit_func
        else:
            self.time_limit = time_limit

        if logger is not None:
            self.logging = logger.info
        else:
            self.logging = print

        self.self_ip = get_host_ip() if self_ip is None else self_ip
        self.self_port = self_port
        self.self_authkey = self_authkey
        self.reporter = Reporter(self.worker_messager, ip=self.self_ip, port=self.self_port,
                                 authkey=self.self_authkey, logger=logger)

    def run(self):
        while True:
            # tell master worker is ready
            init_observation = [None, None, None, None, None]
            try:
                self.worker_messager.send_message(init_observation)
            except Exception as e:
                self.logging("Worker send init message error: %s" % str(e))
                return

            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                self.logging("Worker receive message error: %s" % str(e))
                return
            if msg is None:
                # Wait for configs
                time.sleep(self.sleep_time)
                continue
            self.logging("Worker: get msg: %s. start working." % msg)
            config, extra_conf, time_limit_per_trial, n_iteration, trial_id = msg

            reporter_meta_data = dict(
                config=config,
                trial_id=trial_id,
                extra_conf=extra_conf,
            )
            self.reporter.reset(reporter_meta_data)

            # Start working
            try:
                args, kwargs = (config, n_iteration, extra_conf, self.reporter), dict()
                self.objective_function(*args, **kwargs)    # todo: time_limit
            except StoppingException:   # todo: time_limit
                pass
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
