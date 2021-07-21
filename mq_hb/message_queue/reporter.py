import time
import traceback
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from mq_hb.message_queue.receiver_messager import ReceiverMessager
from mq_hb.utils import StoppingException


class Reporter(object):
    """
    Reporter for stopping variant worker
    Report observation to master and receive stop/continue message
    """
    def __init__(self, worker_messager, ip="127.0.0.1", port=13579, authkey=b'abc', logger=None):
        self.ip = ip
        assert self.ip != "", "ip cannot be \"\" because need to be sent to master!"
        self.port = port
        self.authkey = authkey

        if logger is not None:
            self.logging = logger.info
        else:
            self.logging = print

        # worker messager to report observation to master
        self.worker_messager = worker_messager
        # receiver messager to receive stop/continue message from master
        try:
            self.receiver_messager = ReceiverMessager(ip=ip, port=port, authkey=authkey, max_len=100)
        except Exception:
            self.logging("Reporter init receiver messager error: %s" % traceback.format_exc())
            raise

        self.meta_data = None



    def __call__(self, objective_value, n_iteration, time_taken, test_perf=None, ref_id=None, early_stop=False):
        trial_state = SUCCESS   # todo
        config = self.meta_data['config']
        trial_id = self.meta_data['trial_id']
        extra_conf = self.meta_data['extra_conf']

        return_info = dict(loss=objective_value,
                           n_iteration=n_iteration,
                           ref_id=ref_id,
                           early_stop=early_stop,
                           trial_state=trial_state,
                           test_perf=test_perf,
                           extra_conf=extra_conf)
        worker_info = dict(ip=self.ip, port=self.port, authkey=self.authkey)
        observation = [return_info, time_taken, trial_id, config, worker_info]

        # send observation
        try:
            self.worker_messager.send_message(observation)
            self.logging("Reporter send observation: %s" % str(observation))
        except Exception:
            self.logging("Reporter send observation error: %s" % traceback.format_exc())
            raise

        # receive message
        while True:
            msg = self.receiver_messager.receive_message()
            if msg is None:
                # Wait
                time.sleep(0.1)
                continue
            break

        self.logging("Reporter receive msg: %s" % str(msg))
        next_n_iteration = msg
        if next_n_iteration == -1:
            raise StoppingException
        else:
            return next_n_iteration

    def reset(self, meta_data: dict):
        self.meta_data = meta_data
        for k in ['config', 'trial_id', 'extra_conf']:
            assert k in self.meta_data.keys(), '%s not in meta_data!' % (k, )
        self.clear()

    def clear(self):
        cnt = 0
        while True:
            msg = self.receiver_messager.receive_message()
            if msg is None:
                break
            cnt += 1
            self.logging('Reporter clears msg: %s' % str(msg))
        if cnt > 0:
            self.logging('Reporter clears %d msg. Please check!' % cnt)

    def shutdown(self):
        self.receiver_messager.shutdown()
