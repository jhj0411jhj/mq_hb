import sys
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')

from test.resnet.resnet_obj import mf_objective_func_gpu_stopping, test_config
from mq_hb.utils import StoppingException

R = 27
eta = 3


def ordinary_reporter(**kwargs):
    print('reporter get: %s' % kwargs)
    n_iteration = kwargs['n_iteration']
    if n_iteration == R:
        print('reporter send: stop')
        raise StoppingException
    else:
        ret = n_iteration * eta
        print('reporter send: %d' % ret)
        return ret


if __name__ == '__main__':
    extra_conf = dict()
    mf_objective_func_gpu_stopping(
        config=test_config, n_resource=1, extra_conf=extra_conf, reporter=ordinary_reporter,
        device='cuda', total_resource=R,
    )

