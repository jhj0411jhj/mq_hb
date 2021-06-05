import numpy as np
from nasbench import api


# Load the data from file (this will take some time)
NASBENCH_TFRECORD = './nas_data/nasbench_full.tfrecord'
nasbench = api.NASBench(NASBENCH_TFRECORD)

print('loading finish!!!')  # 259 seconds

print(nasbench.valid_epochs, type(nasbench.valid_epochs))
assert nasbench.valid_epochs == {4, 12, 36, 108}

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

NASBENCH101_REPEAT_NUM = 3


# Create an Inception-like module (5x5 convolution replaced with two 3x3
# convolutions).
model_spec = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
            [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Query this model from dataset, returns a dictionary containing the metrics
# associated with this model.
# print('Querying an Inception-like model.')
# data = nasbench.query(model_spec, epochs=108)
# print(data)
# print(nasbench.get_budget_counters())   # prints (total time, total epochs)


# Get all metrics (all epoch lengths, all repeats) associated with this
# model_spec. This should be used for dataset analysis and NOT for
# benchmarking algorithms (does not increment budget counters).
print('\nGetting all metrics for the same Inception-like model.')
fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
print(fixed_metrics)
print()


epochs = 108

assert len(computed_metrics[epochs]) == NASBENCH101_REPEAT_NUM

train_time = []
val_perf = []
test_perf = []
for repeat_index in range(NASBENCH101_REPEAT_NUM):
    data_point = computed_metrics[epochs][repeat_index]
    train_time.append(data_point['final_training_time'])
    val_perf.append(data_point['final_validation_accuracy'])
    test_perf.append(data_point['final_test_accuracy'])
train_time = np.mean(train_time, dtype=np.float64)
val_perf = np.mean(val_perf, dtype=np.float64)
test_perf = np.mean(test_perf, dtype=np.float64)

print('epochs=%d. mean training_time: %f, validation_accuracy: %f, test_accuracy: %f'
      % (epochs, train_time, val_perf, test_perf))
