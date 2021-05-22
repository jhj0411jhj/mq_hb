from nas_201_api import NASBench201API as API
import time


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def show(self, msg=''):
        new_time = time.time()
        print('===== %.2fs. %s' % (new_time-start_time, msg), flush=True)
        self.start_time = new_time


timer = Timer()

NAS_DATA_PATH = '../nas_data/NAS-Bench-201-v1_1-096897.pth'

start_time = time.time()
api = API(NAS_DATA_PATH, verbose=True)

timer.show('api created!')

num = len(api)
print(num)
# for i, arch_str in enumerate(api):
#     print('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))

# show all information for a specific architecture
api.show(1)
api.show(2)
timer.show('show complete!')

# show the mean loss and accuracy of an architecture
info = api.query_meta_info_by_index(1)  # This is an instance of `ArchResults`
res_metrics = info.get_metrics('cifar10', 'train')  # This is a dict with metric names as keys
cost_metrics = info.get_compute_costs(
    'cifar100')  # This is a dict with metric names as keys, e.g., flops, params, latency

timer.show('query')

# get the detailed information
results = api.query_by_index(1, 'cifar100')  # a dict of all trials for 1st net on cifar100, where the key is the seed
print('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
for seed, result in results.items():
    print('Latency : {:}'.format(result.get_latency()))
    print('Train Info : {:}'.format(result.get_train()))
    print('Valid Info : {:}'.format(result.get_eval('x-valid')))
    print('Test  Info : {:}'.format(result.get_eval('x-test')))
    # for the metric after a specific epoch
    print('Train Info [10-th epoch] : {:}'.format(result.get_train(10)))

timer.show('query and result')

index = api.query_index_by_arch(
    '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
api.show(index)

timer.show('query by arch')

# obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
config = api.get_net_config(123, 'cifar10')
print(config)

timer.show('get config')

