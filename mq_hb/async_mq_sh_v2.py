import time
import numpy as np
from math import log, ceil
from mq_hb.async_mq_base_facade import async_mqBaseFacade
from mq_hb.utils import WAITING, RUNNING, COMPLETED, PROMOTED
from mq_hb.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.constants import MAXINT


class async_mqSuccessiveHalving_v2(async_mqBaseFacade):
    """
    Asynchronous Successive Halving with promotion cycle
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 random_state=1,
                 method_id='mqASHA_new',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        max_queue_len = 3 * R   # conservative design
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

    def create_bracket(self):
        """
        bracket : list of rungs
        rung: {
            'rung_id': rung id (the lowest rung is 0),
            'n_iteration': iterations (resource) per config for evaluation,
            'promotion_cycle': number of configs to activate promotion,
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
            promotion_cycle = int(self.R * self.eta ** (-i))  # int(n * self.eta ** (-i))
            rung = dict(
                rung_id=i,
                n_iteration=n_iteration,
                promotion_cycle=promotion_cycle,
                jobs=list(),
                configs=set(),
                num_promoted=0,
            )
            self.bracket.append(rung)
        self.logger.info('Init bracket: %s.' % str(self.bracket))

    def get_job(self):
        """
        find waiting config, or sample a new config
        """
        next_config = None
        next_n_iteration = None
        next_extra_conf = None
        # find waiting config from top to bottom
        for rung_id in reversed(range(len(self.bracket))):
            for job_id, job in enumerate(self.bracket[rung_id]['jobs']):
                job_status, config, perf, extra_conf = job
                # set running
                if job_status == WAITING:
                    next_config = config
                    next_n_iteration = self.bracket[rung_id]['n_iteration']
                    next_extra_conf = extra_conf
                    # update bracket
                    self.logger.info('Running job in rung %d: %s' % (rung_id, self.bracket[rung_id]['jobs'][job_id]))
                    self.bracket[rung_id]['jobs'][job_id][0] = RUNNING
                    break
            if next_config is not None:
                break

        # no waiting config, sample a new one
        if next_config is None:
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

    def check_promotion(self, rung_id):
        """
        check promotion cycle, then promote (set waiting)
        CAUTION: call this function only after update_observation() every time
        """
        # do not check highest rung
        if rung_id == len(self.bracket) - 1:
            return

        # check whether number of complete jobs reaches promotion cycle
        num_completed_promoted = len([job for job in self.bracket[rung_id]['jobs']
                                      if job[0] in (COMPLETED, PROMOTED)])
        if num_completed_promoted % self.bracket[rung_id]['promotion_cycle'] != 0:
            return

        # job: [job_status, config, perf, extra_conf]
        self.bracket[rung_id]['jobs'] = sorted(self.bracket[rung_id]['jobs'], key=lambda x: x[2])
        completed_jobs = [(job_id, job) for job_id, job in enumerate(self.bracket[rung_id]['jobs'])
                          if job[0] in (COMPLETED, )]

        n_set_promote = 0
        n_should_promote = self.bracket[rung_id + 1]['promotion_cycle']
        candidate_jobs = completed_jobs[0: n_should_promote]
        for job_id, job in candidate_jobs:
            job_status, config, perf, extra_conf = job
            if not perf < MAXINT:
                self.logger.warning('Skip promoting job (bad perf): %s' % job)
                continue
            # check if config already exists in upper rungs
            exist = False
            for i in range(rung_id + 1, len(self.bracket)):
                if config in self.bracket[i]['configs']:
                    exist = True
                    break
            if exist:
                self.logger.warning('Skip promoting job (duplicate): %s' % job)
                continue

            # promote (set waiting)
            n_set_promote += 1
            next_config = config
            next_n_iteration = self.bracket[rung_id + 1]['n_iteration']
            next_extra_conf = extra_conf
            next_extra_conf['initial_run'] = False  # for loading from checkpoint in DL
            # update bracket
            self.logger.info('Promote job in rung %d: %s' % (rung_id, self.bracket[rung_id]['jobs'][job_id]))
            self.bracket[rung_id]['jobs'][job_id][0] = PROMOTED
            self.bracket[rung_id]['num_promoted'] += 1
            new_job = [WAITING, next_config, MAXINT, next_extra_conf]  # running perf is set to MAXINT
            self.bracket[rung_id + 1]['jobs'].append(new_job)
            self.bracket[rung_id + 1]['configs'].add(next_config)
            assert len(self.bracket[rung_id + 1]['jobs']) == len(self.bracket[rung_id + 1]['configs'])

        if n_set_promote != n_should_promote:
            self.logger.warning('In rung %d, promote: %d, should promote: %d.'
                                % (rung_id, n_set_promote, n_should_promote))
        else:
            self.logger.info('In rung %d, promote %d configs.' % (rung_id, n_set_promote))

    def update_observation(self, config, perf, n_iteration):
        """
        update bracket and check promotion cycle
        """
        # update bracket
        rung_id = self.get_rung_id(self.bracket, n_iteration)

        updated = False
        for job in self.bracket[rung_id]['jobs']:
            _job_status, _config, _perf, _extra_conf = job
            if _config == config:
                assert _job_status == RUNNING
                job[0] = COMPLETED
                job[2] = perf
                updated = True
                break
        assert updated
        # print('=== bracket after update_observation:', self.get_bracket_status(self.bracket))

        if int(n_iteration) == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)

        # check promotion cycle
        self.check_promotion(rung_id)
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
        status = '\n' + '=' * 54 + '\n'
        status += 'rung_id n_iteration PROMOTED COMPLETED RUNNING WAITING\n'
        for rung in bracket:
            rung_id = rung['rung_id']
            n_iteration = rung['n_iteration']
            jobs = rung['jobs']
            num_waiting, num_running, num_completed, num_promoted = 0, 0, 0, 0
            for _job_status, _config, _perf, _extra_conf in jobs:
                if _job_status == WAITING:
                    num_waiting += 1
                elif _job_status == RUNNING:
                    num_running += 1
                elif _job_status == COMPLETED:
                    num_completed += 1
                elif _job_status == PROMOTED:
                    num_promoted += 1
            status += '%7d %11d %8d %9d %7d %7d\n' % (rung_id, n_iteration,
                                                      num_promoted, num_completed, num_running, num_waiting)
        status += '=' * 54 + '\n'
        return status

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
