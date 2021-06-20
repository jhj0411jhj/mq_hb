import time
import os
import warnings
from math import ceil, log
import numpy as np
from sklearn.metrics import accuracy_score
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from openbox.utils.constants import MAXINT

from fcnet_util import get_path_by_config
from fcnet_dataset import ImageDataset, SubsetSequentialampler

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

# Constant
input_dims = 784
max_epoch = 40.5
train_size = 48000
image_size = 28
data_dir = './datasets/img_datasets/mnist/'

image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size, grayscale=True)


def get_fcnet_configspace():
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-4, 1e-1, log=True, default_value=5e-3)
    momentum = UniformFloatHyperparameter("momentum", 0., .9, default_value=0., q=.1)
    # lr_decay = UniformFloatHyperparameter("lr_decay", .7, .99, default_value=9e-1)
    n_layer1 = UniformIntegerHyperparameter("n_layer1", 32, 256, default_value=96, q=8)
    n_layer2 = UniformIntegerHyperparameter("n_layer2", 64, 256, default_value=128, q=8)
    batch_size = CategoricalHyperparameter("batch_size", [50, 100, 200], default_value=100)
    dropout1 = UniformFloatHyperparameter("kb_1", .3, .9, default_value=.5, q=.1)
    dropout2 = UniformFloatHyperparameter("kb_2", .3, .9, default_value=.5, q=.1)
    kernel_regularizer = UniformFloatHyperparameter("k_reg", 1e-9, 1e-4, default_value=1e-6, log=True)
    cs.add_hyperparameters([learning_rate, momentum, n_layer1, n_layer2, batch_size, dropout1, dropout2,
                            kernel_regularizer])
    return cs


def mf_objective_func_gpu(config, n_resource, extra_conf, device, total_resource, run_test=True,
                          model_dir='./data/fcnet_save_models/unnamed_trial', eta=3):  # device='cuda' 'cuda:0'
    print('extra_conf:', extra_conf)
    initial_run = extra_conf['initial_run']
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except FileExistsError:
        pass

    _transforms = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    image_data.load_data(_transforms, _transforms)
    start_time = time.time()

    n_layer1 = config['n_layer1']
    n_layer2 = config['n_layer2']
    dropout1 = config['kb_1']
    dropout2 = config['kb_2']
    model = nn.Sequential(nn.Linear(input_dims, n_layer1),
                          nn.ReLU(),
                          nn.Dropout(p=1 - dropout1),
                          nn.Linear(n_layer1, n_layer2),
                          nn.ReLU(),
                          nn.Dropout(p=1 - dropout2),
                          nn.Linear(n_layer2, 10))
    model.to(device)

    epoch_ratio = float(n_resource) / float(total_resource)

    config_model_path = os.path.join(model_dir,
                                     'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource / eta) + '.pt')
    save_path = os.path.join(model_dir,
                             'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource) + '.pt')

    # Continue training if initial_run=False
    if not initial_run:
        if not os.path.exists(config_model_path):
            raise ValueError('not initial_run but config_model_path not exists. check if exists duplicated configs '
                             'and saved model were removed.')
        epoch_num = max_epoch * epoch_ratio - max_epoch * epoch_ratio / eta
        load_path = True

    else:
        load_path = False
        epoch_num = max_epoch * epoch_ratio

    step_num = int(epoch_num * train_size / config['batch_size'])
    print(step_num)
    try:
        params = model.parameters()
        optimizer = SGD(params=params, lr=config['learning_rate'], momentum=config['momentum'],
                        weight_decay=config['k_reg'])
        # scheduler = MultiStepLR(optimizer, milestones=[int(max_epoch * 0.5), int(max_epoch * 0.75)],
        #                         gamma=config['lr_decay'])
        loss_func = nn.CrossEntropyLoss()

        if load_path:
            checkpoint = torch.load(config_model_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            train_sampler = checkpoint['train_sampler']
            cur_step_num = checkpoint['step_num']

        else:
            cur_step_num = 0
            train_sampler = SubsetSequentialampler(image_data.train_indices)

        train_loader = DataLoader(dataset=image_data.train_dataset, batch_size=config['batch_size'],
                                  sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset=image_data.train_for_val_dataset, batch_size=config['batch_size'],
                                sampler=image_data.val_sampler, num_workers=4)

        epoch_avg_loss = 0
        epoch_avg_acc = 0
        num_train_samples = 0

        train_loader_iter = iter(train_loader)
        for step in range(int(cur_step_num), int(cur_step_num) + int(step_num)):
            model.train()
            # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])

            try:
                data = next(train_loader_iter)
            except:
                train_loader = DataLoader(dataset=image_data.train_dataset, batch_size=config['batch_size'],
                                          sampler=train_sampler, num_workers=4)
                train_loader_iter = iter(train_loader)
                data = next(train_loader_iter)

            batch_x, batch_y = data[0], data[1]
            num_train_samples += len(batch_x)
            batch_x = batch_x.view(batch_x.size(0), -1)
            logits = model(batch_x.float().to(device))
            loss = loss_func(logits, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
            prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
            epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

            if (step + 1) % 120 == 0:
                epoch_avg_loss /= num_train_samples
                epoch_avg_acc /= num_train_samples
                print('Step %d: Train loss %.4f, train acc %.4f' % (step + 1, epoch_avg_loss, epoch_avg_acc))
                num_train_samples = 0
                epoch_avg_loss = 0
                epoch_avg_acc = 0

            if (step + 1) % 240 == 0:
                model.eval()
                with torch.no_grad():
                    num_val_samples = 0
                    val_avg_acc = 0
                    val_avg_loss = 0
                    for i, data in enumerate(val_loader):
                        batch_x, batch_y = data[0], data[1]
                        batch_x = batch_x.view(batch_x.size(0), -1)
                        logits = model(batch_x.float().to(device))
                        val_loss = loss_func(logits, batch_y.to(device))
                        num_val_samples += len(batch_x)
                        val_avg_loss += val_loss.to('cpu').detach() * len(batch_x)

                        prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                        val_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)
                    val_avg_loss /= num_val_samples
                    val_avg_acc /= num_val_samples
                    print('Step %d: Val loss %.4f, val acc %.4f' % (step + 1, val_avg_loss, val_avg_acc))

            # scheduler.step()

        score = val_avg_acc
        optimizer_ = optimizer
        step_num_ = int(step_num) + int(cur_step_num)
        # scheduler_ = scheduler
        train_sampler_ = train_loader.sampler

    except Exception as e:
        import traceback
        traceback.print_exc()
        score = -MAXINT
    print('Evaluation | Score: %.4f | Time cost: %.2f seconds' %
          (score,
           time.time() - start_time))
    print(str(config))

    # Save low-resource models
    if np.isfinite(score) and epoch_ratio != 1.0:
        state = {'model': model.state_dict(),
                 'optimizer': optimizer_.state_dict(),
                 # 'scheduler': scheduler_.state_dict(),
                 'train_sampler': train_sampler_,
                 'step_num': step_num_}
        torch.save(state, save_path)

    try:
        if epoch_ratio == 1:
            s_max = int(log(total_resource) / log(eta))
            for i in range(0, s_max + 1):
                if os.path.exists(os.path.join(model_dir,
                                               'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt')):
                    os.remove(os.path.join(model_dir,
                                           'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt'))
    except Exception as e:
        print('unexpected exception!')
        import traceback
        traceback.print_exc()

    # if np.isfinite(score):
    #     save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score)
    #     if save_flag is True:
    #         state = {'model': estimator.model.state_dict(),
    #                  'optimizer': estimator.optimizer_.state_dict(),
    #                  'scheduler': estimator.scheduler.state_dict(),
    #                  'epoch_num': estimator.epoch_num,
    #                  'early_stop': estimator.early_stop}
    #         torch.save(state, model_path)
    #         print("Model saved to %s" % model_path)
    #
    #     # In case of double-deletion
    #     try:
    #         if delete_flag and os.path.exists(model_path_deleted):
    #             os.remove(model_path_deleted)
    #             print("Model deleted from %s" % model_path)
    #     except:
    #         pass

    # Turn it into a minimization problem.
    result = dict(
        objective_value=-score,
    )
    return result


def dl_holdout_validation(estimator, scorer, dataset, random_state=1, **kwargs):
    start_time = time.time()
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        estimator.fit(dataset, **kwargs)
        if 'profile_epoch' in kwargs or 'profile_iter' in kwargs:
            return time.time() - start_time
        else:
            return scorer._sign * estimator.score(dataset, scorer._score_func)


if __name__ == '__main__':
    cs = get_fcnet_configspace()
    test_config = cs.get_default_configuration()
    extra_conf = dict(initial_run=True)
    result = mf_objective_func_gpu(config=test_config, n_resource=27, extra_conf=extra_conf, device='cuda', total_resource=81)
    print(result)
