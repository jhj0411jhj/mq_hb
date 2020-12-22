import time
import sys
import traceback
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException
from litebo.core.message_queue.worker_messager import WorkerMessager


class mqmfWorker(object):
    """
    message queue worker for multi-fidelity optimization
    """
    def __init__(self, objective_function, ip="127.0.0.1", port=13579):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port)

    def run(self):
        while True:
            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                print("Worker receive message error:", str(e))
                return
            if msg is None:
                # Wait for configs
                time.sleep(1)
                continue
            print("Worker: get config. start working.")
            config, time_limit_per_trial, n_iteration, trail_id = msg

            # Start working
            start_time = time.time()
            trial_state = SUCCESS
            try:
                args, kwargs = (config, n_iteration), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
                else:
                    perf = _result if _result is not None else MAXINT
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                perf = MAXINT

            time_taken = time.time() - start_time
            return_info = dict(loss=perf,
                               ref_id=579,
                               early_stop=False,
                               trail_state=trial_state)   # todo ref_id, early_stop
            observation = [return_info, time_taken, trail_id, config]

            # Send result
            print("Worker: perf=%f. sending result." % perf)
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                print("Worker send message error:", str(e))
                return
