from typing import List
try:
    from litebo.config_space import Configuration, ConfigurationSpace
except ImportError as e:
    from litebo.utils.config_space import Configuration, ConfigurationSpace


# TODO: escape the bug.
def sample_configurations(configuration_space: ConfigurationSpace, num: int) -> List[Configuration]:
    result = []
    cnt = 0
    while cnt < num:
        config = configuration_space.sample_configuration(1)
        if config not in result:
            result.append(config)
            cnt += 1
    return result


def expand_configurations(configs: List[Configuration], configuration_space: ConfigurationSpace, num: int):
    num_config = len(configs)
    num_needed = num - num_config
    config_cnt = 0
    while config_cnt < num_needed:
        config = configuration_space.sample_configuration(1)
        if config not in configs:
            configs.append(config)
            config_cnt += 1
    return configs
