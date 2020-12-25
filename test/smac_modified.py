import time as time_pkg
from smac.runhistory.runhistory import *


class RunHistory_modified(RunHistory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_start_time = time_pkg.time()
        self.global_trial_counter = 0
        self.exp_recorder = []

    def add(self, config: Configuration, cost: float, time: float,
            status: StatusType, instance_id: str = None,
            seed: int = None,
            additional_info: dict = None,
            origin: DataOrigin = DataOrigin.INTERNAL):
        # save record
        global_time = time_pkg.time() - self.global_start_time
        trial_id = self.global_trial_counter
        self.global_trial_counter += 1
        return_info = dict(loss=cost,
                           trial_state=status)
        record = {'trial_id': trial_id, 'time_consumed': time,
                  'configuration': config,
                  'return_info': return_info, 'global_time': global_time}
        self.exp_recorder.append(record)
        print('smac add record:', record, flush=True)
        # super class add
        super().add(config, cost, time, status, instance_id, seed, additional_info, origin)
