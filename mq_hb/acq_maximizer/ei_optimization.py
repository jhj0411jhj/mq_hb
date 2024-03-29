import numpy as np
from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.utils.constants import MAXINT

from mq_hb.surrogate.mf_gp import convert_configurations_to_resource_array


class RandomSampling(object):

    def __init__(self, acquisition_function, config_space, n_samples=5000, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        acquisition_function:
            The acquisition function which will be maximized
        n_samples: int
            Number of candidates that are samples
        """
        self.config_space = config_space
        self.acquisition_function = acquisition_function
        if rng is None:
            self.rng = np.random.RandomState(1357)
        else:
            self.rng = rng
        self.n_samples = n_samples

    def maximize(self, best_config, batch_size=1):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------
        batch_size: number of maximizer returned.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        incs_configs = list(get_one_exchange_neighbourhood(best_config, seed=self.rng.randint(MAXINT)))

        # Sample random points uniformly over the whole space
        # rand_configs = self.config_space.sample_configuration(max(self.n_samples, batch_size) - len(incs_configs))
        rand_configs = self.config_space.sample_configuration(max(self.n_samples, batch_size))

        configs_list = incs_configs + rand_configs

        y = self.acquisition_function(configs_list)
        y = y.reshape(-1)
        assert y.shape[0] == len(configs_list)

        candidates = [configs_list[int(i)] for i in np.argsort(-y)[:batch_size]]   # maximize
        return candidates


class mf_RandomSampling(object):

    def __init__(self, acquisition_function, config_space, max_resource, log_scale=True, n_samples=5000, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        acquisition_function:
            The acquisition function which will be maximized
        n_samples: int
            Number of candidates that are samples
        log_scale: bool
            Convert log scale resource.
        """
        self.config_space = config_space
        self.acquisition_function = acquisition_function
        if rng is None:
            self.rng = np.random.RandomState(1357)
        else:
            self.rng = rng
        self.n_samples = n_samples
        self.max_resource = max_resource
        self.log_scale = log_scale

    def maximize(self, resource, best_config, batch_size=1):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------
        resource:

        best_config:

        batch_size: number of maximizer returned.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        incs_configs = list(get_one_exchange_neighbourhood(best_config, seed=self.rng.randint(MAXINT)))

        # Sample random points uniformly over the whole space
        # rand_configs = self.config_space.sample_configuration(max(self.n_samples, batch_size) - len(incs_configs))
        rand_configs = self.config_space.sample_configuration(max(self.n_samples, batch_size))

        configs_list = incs_configs + rand_configs
        resource_list = [resource] * len(configs_list)
        config_array = convert_configurations_to_resource_array(configs_list, resource_list, self.max_resource,
                                                                log_scale=self.log_scale)

        y = self.acquisition_function(config_array, convert=False)
        y = y.reshape(-1)
        assert y.shape[0] == len(configs_list)

        candidates = [configs_list[int(i)] for i in np.argsort(-y)[:batch_size]]   # maximize
        return candidates
