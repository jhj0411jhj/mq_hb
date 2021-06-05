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

"""
===== iepoch: 0
{'train-loss': 1.8142211441802978, 'train-accuracy': 31.002666658121743, 
'train-per-time': 10.848206877708435, 'train-all-time': 10.848206877708435, 
'valid-loss': 1.9344430942281086, 'valid-accuracy': 34.5200000012207, 
'valid-per-time': 3.754198976925442, 'valid-all-time': 3.754198976925442}

===== iepoch: 1
{'train-loss': 1.388922302246094, 'train-accuracy': 48.423999997965495, 
'train-per-time': 10.848206877708435, 'train-all-time': 21.69641375541687, 
'valid-loss': 1.7040676763153078, 'valid-accuracy': 44.37733333333333, 
'valid-per-time': 3.754198976925442, 'valid-all-time': 7.508397953850884}
"""

