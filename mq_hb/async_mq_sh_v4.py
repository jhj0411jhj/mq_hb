import time
import numpy as np
from math import log, ceil
from mq_hb.async_mq_base_facade import async_mqBaseFacade
from mq_hb.utils import RUNNING, COMPLETED, PROMOTED
from mq_hb.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.constants import MAXINT


class async_mqSuccessiveHalving_v4(async_mqBaseFacade):
    """
    The implementation of Asynchronous Successive Halving Algorithm (ASHA)
    v0: origin version
    v4: use sh at first to perform delay. (bad coding, just for test)
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 sh_round=2,
                 random_state=1,
                 method_id='mqASHA',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        max_queue_len = 1000   # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)

        self.R = R      # Maximum iterations per configuration
        self.eta = eta  # Define configuration downsampling rate (default = 3)
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))

        self.incumbent_configs = list()
        self.incumbent_perfs = list()

        self.bracket = None
        self.create_bracket()

        self.sh_round = sh_round    # how many sh to do before starting asha
        self.start_asha = False

    def create_bracket(self):
        """
        bracket : list of rungs
        rung: {
            'rung_id': rung id (the lowest rung is 0),
            'n_iteration': iterations (resource) per config for evaluation,
            'jobs': list of [job_status, config, perf, extra_conf],
            'configs': set of all configs in the rung,
            'num_promoted': number of promoted configs in the rung,
        }
        job_status: RUNNING, COMPLETED, PROMOTED
        """
        self.bracket = list()
        s = self.s_max
        # Initial number of iterations per config
        r = self.R * self.eta ** (-s)
        for i in range(s + 1):
            n_iteration = r * self.eta ** (i)
            promotion_start_threshold = self.R * self.eta ** (-i)
            rung = dict(
                rung_id=i,
                n_iteration=n_iteration,
                jobs=list(),
                configs=set(),
                num_promoted=0,
                # set promotion start threshold
                promotion_start_threshold=promotion_start_threshold,
                v=promotion_start_threshold,    # to accumulate
            )
            self.bracket.append(rung)
        self.logger.info('Init bracket: %s.' % str(self.bracket))

    def get_job(self):
        """
        find promotable config, or sample a new config
        """
        next_config = None
        next_n_iteration = None
        next_extra_conf = None
        # find promotable config from top to bottom
        for rung_id in reversed(range(len(self.bracket) - 1)):
            # if not enough jobs, do not promote
            if not self.can_promote(rung_id):
                continue

            # job: [job_status, config, perf, extra_conf]
            self.bracket[rung_id]['jobs'] = sorted(self.bracket[rung_id]['jobs'], key=lambda x: x[2])
            complete_jobs = [(job_id, job) for job_id, job in enumerate(self.bracket[rung_id]['jobs'])
                             if job[0] in (COMPLETED, PROMOTED)]

            # keep the first 1/eta
            candidate_jobs = complete_jobs[0: int(len(complete_jobs) / self.eta)]
            for job_id, job in candidate_jobs:
                job_status, config, perf, extra_conf = job
                if not (job_status == COMPLETED and perf < MAXINT):
                    continue
                # check if config already exists in upper rungs
                exist = False
                for i in range(rung_id + 1, len(self.bracket)):
                    if config in self.bracket[i]['configs']:
                        exist = True
                        break
                # promote
                if not exist:
                    next_config = config
                    next_n_iteration = self.bracket[rung_id + 1]['n_iteration']
                    next_extra_conf = extra_conf.copy()
                    next_extra_conf['initial_run'] = False  # for loading from checkpoint in DL
                    # update bracket
                    self.logger.info('Promote job in rung %d: %s' % (rung_id, self.bracket[rung_id]['jobs'][job_id]))
                    self.bracket[rung_id]['jobs'][job_id][0] = PROMOTED
                    self.bracket[rung_id]['num_promoted'] += 1
                    new_job = [RUNNING, next_config, MAXINT, next_extra_conf]     # running perf is set to MAXINT
                    self.bracket[rung_id + 1]['jobs'].append(new_job)
                    self.bracket[rung_id + 1]['configs'].add(next_config)
                    assert len(self.bracket[rung_id + 1]['jobs']) == len(self.bracket[rung_id + 1]['configs'])
                    break
            if next_config is not None:
                break

        if next_config is not None:
            return next_config, next_n_iteration, next_extra_conf

        # do not sample new config, if in sh round
        if len(self.bracket[0]['jobs']) >= self.bracket[0]['promotion_start_threshold'] and not self.start_asha:
            next_config = self.bracket[0]['jobs'][0][1]     # use evaluated config to simulate waiting
            next_n_iteration = 1
            next_extra_conf = {}
            next_extra_conf['initial_run'] = True  # for loading from checkpoint in DL
            self.logger.info('simulate waiting. do not sample new config.')
            time.sleep(3)
            return next_config, next_n_iteration, next_extra_conf

        # no promotable config, sample a new one
        elif next_config is None:
            next_config, next_n_iteration, next_extra_conf = self.choose_next()
            next_extra_conf['initial_run'] = True  # for loading from checkpoint in DL
            # update bracket
            rung_id = self.get_rung_id(self.bracket, next_n_iteration)
            self.logger.info('Sample a new config: %s. Add to rung %d.' % (next_config, rung_id))
            new_job = [RUNNING, next_config, MAXINT, next_extra_conf]   # running perf is set to MAXINT
            self.bracket[rung_id]['jobs'].append(new_job)
            self.bracket[rung_id]['configs'].add(next_config)
            assert len(self.bracket[rung_id]['jobs']) == len(self.bracket[rung_id]['configs'])

        # print('=== bracket after get_job:', self.get_bracket_status(self.bracket))
        return next_config, next_n_iteration, next_extra_conf

    def can_promote(self, rung_id):
        """
        return whether configs can be promoted in current rung
        """
        if self.start_asha:     # second stage: origin asha
            return True

        # first stage: sh
        # if not enough jobs, do not promote
        num_completed_promoted = len([job for job in self.bracket[rung_id]['jobs']
                                      if job[0] in (COMPLETED, PROMOTED)])
        num_promoted = self.bracket[rung_id]['num_promoted']
        if num_completed_promoted == 0 or (num_promoted + 1) / num_completed_promoted > 1 / self.eta:
            return False

        # prevent error promotion in start stage
        promotion_start_threshold = self.bracket[rung_id]['promotion_start_threshold']
        if num_completed_promoted < promotion_start_threshold:
            return False

        return True

    def update_observation(self, config, perf, n_iteration):
        rung_id = self.get_rung_id(self.bracket, n_iteration)

        updated = False
        for job in self.bracket[rung_id]['jobs']:
            _job_status, _config, _perf, _extra_conf = job
            if _config == config:
                if _job_status != RUNNING:
                    return  # ignore redundant config
                job[0] = COMPLETED
                job[2] = perf
                updated = True
                break
        assert updated
        # print('=== bracket after update_observation:', self.get_bracket_status(self.bracket))

        if int(n_iteration) == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)

            if len(self.bracket[rung_id]['jobs']) < self.sh_round:
                for rung in self.bracket:
                    rung['promotion_start_threshold'] += rung['v']
                self.logger.info('Start next sh. full observation num: %d.' % (len(self.bracket[rung_id]['jobs']),))
            else:
                self.start_asha = True
                self.logger.info('Start asha. full observation num: %d.' % (len(self.bracket[rung_id]['jobs']),))
            self.logger.info('=== bracket after update_observation: %s', self.get_bracket_status(self.bracket))

        return

    def choose_next(self):
        """
        sample a random config and give the least iterations
        """
        next_config = sample_configuration(self.config_space, excluded_configs=self.bracket[0]['configs'])
        next_n_iteration = self.bracket[0]['n_iteration']
        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    @staticmethod
    def get_rung_id(bracket, n_iteration):
        rung_id = None
        for rung in bracket:
            if rung['n_iteration'] == n_iteration:
                rung_id = rung['rung_id']
                break
        assert rung_id is not None
        return rung_id

    @staticmethod
    def get_bracket_status(bracket):
        status = '\n' + '=' * 46 + '\n'
        status += 'rung_id n_iteration PROMOTED COMPLETED RUNNING\n'
        for rung in bracket:
            rung_id = rung['rung_id']
            n_iteration = rung['n_iteration']
            jobs = rung['jobs']
            num_running, num_completed, num_promoted = 0, 0, 0
            for _job_status, _config, _perf, _extra_conf in jobs:
                if _job_status == RUNNING:
                    num_running += 1
                elif _job_status == COMPLETED:
                    num_completed += 1
                elif _job_status == PROMOTED:
                    num_promoted += 1
            status += '%7d %11d %8d %9d %7d\n' % (rung_id, n_iteration, num_promoted, num_completed, num_running)
        status += '=' * 46 + '\n'
        return status

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
