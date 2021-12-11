"""
example cmdline:

python test/awd_lstm_lm/test_lstm_best.py --mth amfesv20-n4 --rep 5 --start_id 0

"""

import os
import time
import argparse
import numpy as np
import pickle as pkl

import sys
sys.path.insert(0, '.')
sys.path.insert(1, '../open-box')    # for dependency
from test.utils import timeit
from test.utils import seeds
from lstm_obj import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='penn')
parser.add_argument('--mth', type=str, default='amfesv20-n4')
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--runtime_limit', type=int, default=172800)

args = parser.parse_args()
dataset = args.dataset
mth = args.mth
rep = args.rep
start_id = args.start_id
runtime_limit = args.runtime_limit


def lstm_objective_func_gpu(config, corpus, device='cuda'):  # device='cuda' 'cuda:0'
    t0 = time.time()
    assert corpus is not None

    device = torch.device(device)

    criterion = None
    dropout = config['dropout']
    dropouth = config['dropouth']
    dropouti = config['dropouti']
    dropoute = config['dropoute']
    wdrop = config['wdrop']
    emsize = config['emsize']
    hidden_size = config['hidden_size']
    weight_decay = config['wdecay']
    batch_size = config['batch_size']
    lr = config['lr']
    print('worker receive config:', config)

    epoch_ratio = 1

    ntokens = len(corpus.dictionary)
    model = RNNModel('LSTM', ntokens, emsize, hidden_size, n_layers, dropout, dropouth,
                     dropouti, dropoute, wdrop, tied)

    train_data = batchify(corpus.train, batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)

    init_epoch_num = 1
    epoch_num = ceil(max_epoch * epoch_ratio)
    print('epoch_num', epoch_num)
    ###
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using splits:', splits)
        criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)
    ###

    model = model.to(device)
    criterion = criterion.to(device)
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)
    # Loop over epochs.
    best_val_loss = []
    stored_loss = 100000000
    evals_result = []

    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

    return_pp = 1e10
    for epoch in range(init_epoch_num, init_epoch_num + epoch_num):
        epoch_start_time = time.time()
        train(corpus, model, criterion, optimizer, epoch, batch_size, train_data, bptt)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(corpus, model, criterion, val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            for prm in model.parameters():
                if prm in tmp:
                    prm.data = tmp[prm].clone()
            return_pp = math.exp(val_loss2)

        else:
            val_loss = evaluate(corpus, model, criterion, val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if epoch in decay_epoch:
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)
            return_pp = math.exp(val_loss)

        evals_result.append(return_pp)

    perf = return_pp
    test_perf = None
    train_time = time.time() - t0

    return perf, test_perf, evals_result, train_time


data_path = './test/awd_lstm_lm/data/penn'
corpus = get_corpus(data_path)


print('===== start test %s %s: rep=%d' % (mth, dataset, rep, ))
for i in range(start_id, start_id + rep):
    seed = seeds[i]

    dir_path = 'data/benchmark_lstm/%s-%d/%s/' % (dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('incumbent_new_record_%s-%s-%d-' % (mth, dataset, seed)) \
                and file.endswith('.pkl'):
            # load config
            with open(os.path.join(dir_path, file), 'rb') as f:
                record = pkl.load(f)
            print(dataset, mth, seed, 'loaded!', record, flush=True)

            # run test
            config = record['configuration']
            print('=== lstm best param ===')
            perf, test_perf, evals_result, train_time = lstm_objective_func_gpu(config, corpus)
            print(evals_result)
            print('=== perf(val, test):', perf, test_perf)
            print('=== train time(s):', train_time)
            # print(list(evals_result['validation_0'].values())[0])

            save_item = (config, perf, test_perf, evals_result, train_time)

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            method_id = mth + '-%s-%d-%s' % (dataset, seed, timestamp)
            save_dir_path = 'data/default_test/lstm-%s/' % (dataset, )
            save_file_name = 'lstm_best-%s.pkl' % (method_id,)
            try:
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
            except FileExistsError:
                pass
            with open(os.path.join(save_dir_path, save_file_name), 'wb') as f:
                pkl.dump(save_item, f)
            print('save to:', save_dir_path, save_file_name)
