from nas_201_api import NASBench201API as API
import time

NAS_DATA_PATH = '../nas_data/NAS-Bench-201-v1_1-096897.pth'

start_time = time.time()
api = API(NAS_DATA_PATH, verbose=True)

dataset = 'cifar10-valid'
arch = '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'

for iepoch in range(5):
    info = api.get_more_info(arch, dataset, iepoch=iepoch, hp='200', is_random=False)
    print('===== iepoch:', iepoch)
    print(info)
