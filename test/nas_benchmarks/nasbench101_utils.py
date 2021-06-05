import time
import numpy as np
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import CategoricalHyperparameter
from nasbench import api

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

VALID_EPOCHS = {4, 12, 36, 108}
EDGE_NUM = 21
OP_NUM = 5
NASBENCH101_REPEAT_NUM = 3


def load_nasbench101(path='../nas_data/nasbench_full.tfrecord'):
    s = time.time()
    # Load the data from file (this will take some time)
    nasbench = api.NASBench(path)
    print('nasbench-101 load time: %.2fs' % (time.time() - s))
    assert nasbench.valid_epochs == VALID_EPOCHS
    return nasbench


def get_nasbench101_configspace():
    cs = ConfigurationSpace()
    for i in range(EDGE_NUM):
        cs.add_hyperparameter(CategoricalHyperparameter('edge%d' % i, choices=[0, 1], default_value=0))

    op_list = [CONV3X3, CONV1X1, MAXPOOL3X3]
    for i in range(OP_NUM):
        cs.add_hyperparameter(CategoricalHyperparameter('op%d' % i, choices=op_list, default_value=CONV3X3))
    return cs


EDGE_POS_LIST = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 6),
    (5, 6),
]


def convert_config_to_modelspec(config):
    # Adjacency matrix of the module
    matrix = [[0, 0, 0, 0, 0, 0, 0],  # input layer
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]  # output layer
    for i in range(EDGE_NUM):
        edge = config['edge%d' % i]
        pos = EDGE_POS_LIST[i]
        matrix[pos[0]][pos[1]] = edge

    # Operations at the vertices of the module, matches order of matrix
    ops = [INPUT] + [config['op%d' % i] for i in range(OP_NUM)] + [OUTPUT]

    # Create an Inception-like module
    model_spec = api.ModelSpec(matrix=matrix, ops=ops)
    return model_spec


def parse_metric(computed_metrics, epochs):
    if len(computed_metrics[epochs]) != NASBENCH101_REPEAT_NUM:
        print('WARNING: error repeat num:%d', len(computed_metrics[epochs]))
    train_time = []
    val_perf = []
    test_perf = []
    for repeat_index in range(len(computed_metrics[epochs])):
        data_point = computed_metrics[epochs][repeat_index]
        train_time.append(data_point['final_training_time'])
        val_perf.append(data_point['final_validation_accuracy'])
        test_perf.append(data_point['final_test_accuracy'])
    train_time = np.mean(train_time, dtype=np.float64)
    val_perf = np.mean(val_perf, dtype=np.float64)
    test_perf = np.mean(test_perf, dtype=np.float64)
    return train_time, val_perf, test_perf


def objective_func(config, n_resource, extra_conf, total_resource, eta, nasbench):
    assert total_resource == 27 and eta == 3
    print('objective extra conf:', extra_conf)
    epochs = int(108 * n_resource / total_resource)
    assert epochs in VALID_EPOCHS, 'error epochs %d' % epochs

    # convert config to modelspec
    model_spec = convert_config_to_modelspec(config)

    # Get all metrics (all epoch lengths, all repeats) associated with this model_spec
    try:
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
    except api.OutOfDomainError as e:
        print('OutOfDomainError: %s' % str(e))
        result = dict(
            objective_value=-0.0,   # minimize
            test_perf=-0.0,         # minimize
            elapsed_time=0.1,
        )
        return result

    train_time, val_perf, test_perf = parse_metric(computed_metrics, epochs)

    # Get checkpoint metrics
    if epochs == 4:
        last_train_time = 0.0
    else:
        last_epochs = int(epochs / eta)
        last_train_time, _, _ = parse_metric(computed_metrics, last_epochs)

    # restore from checkpoint
    train_time = train_time - last_train_time

    result = dict(
        objective_value=-val_perf,  # minimize
        test_perf=-test_perf,       # minimize
        elapsed_time=train_time,
    )
    return result


if __name__ == '__main__':
    cs = get_nasbench101_configspace()
    for i in range(3):
        conf = cs.sample_configuration()
        print(conf)

    test_load = False
    if test_load:
        nasbench = load_nasbench101(path='./nas_data/nasbench_full.tfrecord')
        conf = cs.sample_configuration()
        print(conf)
        result = objective_func(conf, n_resource=3, extra_conf={}, total_resource=27, eta=3, nasbench=nasbench)
        print(result)
