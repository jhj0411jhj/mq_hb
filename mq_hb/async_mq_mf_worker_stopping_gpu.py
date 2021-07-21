from functools import partial
from mq_hb.async_mq_mf_worker_stopping import async_mqmfWorker_stopping


class async_mqmfWorker_stopping_gpu(async_mqmfWorker_stopping):
    """
    async message queue worker for multi-fidelity optimization
    gpu version: specify 'device'
    stopping variant
    """
    def __init__(self, objective_function,
                 device,
                 ip="127.0.0.1", port=13579, authkey=b'abc',
                 self_ip=None, self_port=13531, self_authkey=b'abc',
                 sleep_time=0.1,
                 no_time_limit=True,  # todo: time_limit
                 logger=None):
        objective_function = partial(objective_function, device=device)
        super().__init__(
            objective_function=objective_function,
            ip=ip, port=port, authkey=authkey,
            self_ip=self_ip, self_port=self_port, self_authkey=self_authkey,
            sleep_time=sleep_time,
            no_time_limit=no_time_limit,
            logger=logger
        )
        self.logging('Worker device: %s' % device)
