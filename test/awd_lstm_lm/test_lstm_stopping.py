import sys
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')

from test.awd_lstm_lm.lstm_obj import *
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
    data_path = './test/awd-lstm-lm/data/penn'
    cs = get_lstm_configspace()
    default_config = cs.get_default_configuration()
    corpus = get_corpus(data_path)
    extra_conf = dict()
    result = mf_objective_func_gpu_stopping(default_config, 1, extra_conf, ordinary_reporter,
                                            device='cuda:0', total_resource=R,
                                            model_dir=None,
                                            eta=eta, corpus=corpus)
    print(result)

