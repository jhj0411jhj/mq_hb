import time
import numpy as np
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import CategoricalHyperparameter
import contextlib
import sys
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')

def f(n,start, k):
    w=0.5/(0.5+np.e**(-(n-start)*k))
    return w
s = 10
k = 0.03
a = 0.5
for n in range(0, 100, 5):
    new_last_weight = a / (a + np.e ** (-(n - s) * k))
    print(n, new_last_weight)
exit()

k=0.025
start=10
for i in range(0, 100, 5):
    print(i, f(i, start, k))

exit()


# timer tool
@contextlib.contextmanager
def timeit(name=''):
    print("[%s]Start." % name, flush=True)
    start = time.time()
    yield
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("[%s]Total time = %d hours, %d minutes, %d seconds." % (name, h, m, s), flush=True)

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

VALID_EPOCHS = {4, 12, 36, 108}
MAX_EPOCHS = 108
EDGE_NUM = 21
OP_NUM = 5
NASBENCH101_REPEAT_NUM = 3

def get_nasbench101_configspace():
    cs = ConfigurationSpace()
    for i in range(EDGE_NUM):
        cs.add_hyperparameter(CategoricalHyperparameter('edge%d' % i, choices=[0, 1], default_value=0))

    op_list = [CONV3X3, CONV1X1, MAXPOOL3X3]
    for i in range(OP_NUM):
        cs.add_hyperparameter(CategoricalHyperparameter('op%d' % i, choices=op_list, default_value=CONV3X3))
    return cs


cs = get_nasbench101_configspace()
with timeit('sample'):
    base_configs = cs.sample_configuration(500000)
#print(type(base_configs), len(base_configs), len(set(base_configs)))
with timeit('sample2'):
    new_configs = cs.sample_configuration(5000)
# with timeit('check'):
#     cnt = 0
#     for conf in new_configs:
#         if conf not in base_configs:
#             cnt += 1
#     print('new cnt:', cnt)

with timeit('set'):
    base_configs_set = set(base_configs)
with timeit('check1'):
    cnt = 0
    for conf in new_configs:
        if conf not in base_configs_set:
            cnt += 1
    print('new cnt:', cnt)

with timeit('check2'):
    cnt = 0
    nlist = set(new_configs) - set(base_configs)
    print('len nlist', len(nlist))
    for conf in new_configs:
        if conf in nlist:
            cnt += 1
    print('new cnt2:', cnt)


