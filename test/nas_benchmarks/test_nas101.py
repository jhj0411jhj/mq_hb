# from absl import app
from nasbench import api

# Load the data from file (this will take some time)
NASBENCH_TFRECORD = './nas_data/nasbench_full.tfrecord'
nasbench = api.NASBench(NASBENCH_TFRECORD)

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

print('loading finish!!!')  # 259 seconds

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
print('Querying an Inception-like model.')
data = nasbench.query(model_spec, epochs=108)
print(data)
print(nasbench.get_budget_counters())   # prints (total time, total epochs)


# Get all metrics (all epoch lengths, all repeats) associated with this
# model_spec. This should be used for dataset analysis and NOT for
# benchmarking algorithms (does not increment budget counters).
print('\nGetting all metrics for the same Inception-like model.')
fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
print(fixed_metrics)
for epochs in nasbench.valid_epochs:
    for repeat_index in range(len(computed_metrics[epochs])):
        data_point = computed_metrics[epochs][repeat_index]
        print('Epochs trained %d, repeat number: %d' % (epochs, repeat_index + 1))
        print(data_point)

# Iterate through unique models in the dataset. Models are unqiuely identified
# by a hash.
print('\nIterating over unique models in the dataset.')
for unique_hash in nasbench.hash_iterator():
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(
        unique_hash)
    print(fixed_metrics)

    # For demo purposes, break here instead of iterating through whole set.
    break

