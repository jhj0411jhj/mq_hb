from typing import List
from litebo.config_space import Configuration, ConfigurationSpace


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

