import os.path

import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch
from module.loss.logcosh import LogCosh
from scripts.load_data import load_all_data, resample
from scripts.data_generator import data_regularization, get_mean_stds, WeatherDataset
from procedure.train import BasicTrainer
from scripts.seedInitializer import randomSeedInitial
import logging
from tensorboard_logger import Logger


def dict2str(param: dict):
    result = ''
    for k, v in param.items():
        result.join(f'<{k}, {v}>\t')
    return result


def get_loss_func(loss_type='MAE'):
    if loss_type == 'MAE':
        return nn.L1Loss(), {'MSE': nn.MSELoss()}
    elif loss_type == 'LogCosh':
        return LogCosh(), {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    elif loss_type == 'Huber':
        return nn.SmoothL1Loss(), {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    else:
        raise ValueError(loss_type)


if __name__ == '__main__':
    randomSeedInitial(42)
    experiment = {
        'root': 'logs',
        'model': 'Transformer',
        'bidirectional': True,
        'hidden_size': 32,
        'num_layers': 1,
        'dropout': 0.2,
        'loss': 'Huber',
        'n_epoch': 30
    }
    logs_id = 1
    experiment_keys = ['root', 'model', 'hidden_size', 'bidirectional', 'num_layers', 'dropout', 'n_epoch', 'loss']
    n_epoch = experiment['n_epoch']
    directory = experiment['root']
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for i in range(1, len(experiment_keys)):
        directory = os.path.join(directory, f'{experiment_keys[i]}_{str(experiment[experiment_keys[i]])}')
        if not os.path.isdir(directory):
            os.mkdir(directory)
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(directory, f'logs_{logs_id}.log')),
            logging.StreamHandler()
        ])
    print = logging.info
    tb_logger = Logger(logdir=directory)
    use_col = 15
    look_back = 240
    delay = 24
    name, dataset = resample(load_all_data(root_dir='dataset', use_col=use_col), use_col=use_col)
    dataset = torch.FloatTensor(dataset).t()
    print(dataset.shape)
    train_size = dataset.shape[0] // 2
    val_size = train_size // 2
    test_size = dataset.shape[0] - train_size - val_size
    print(f'{train_size}, {val_size}, {test_size}')
    m, s = get_mean_stds(ds=dataset[: train_size])
    data_regularization(dataset, m, s)
    train_dataset = WeatherDataset(data=dataset, max_index=train_size, shuffle=True, look_back=look_back, delay=delay)
    val_dataset = WeatherDataset(data=dataset, min_index=train_size, max_index=train_size + val_size, shuffle=False,
                                 look_back=look_back, delay=delay)
    test_dataset = WeatherDataset(data=dataset, min_index=train_size + val_size, shuffle=False,
                                  look_back=look_back, delay=delay)
    train_dataset = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_dataset = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False)
    loss_fn, evaluation = get_loss_func(experiment['loss'])
    if experiment['model'] == 'gru':
        from module.models.gru import GRU

        m = GRU(input_size=use_col - 1, length_prediction=delay, num_layers=experiment['num_layers'],
                hidden_size=experiment['hidden_size'], bidirectional=experiment['bidirectional'],
                dropout=experiment['dropout'])
    elif experiment['model'] == 'AttentionGRUV1':
        from module.models.self_attention_gru import AttentionGRUV1

        m = AttentionGRUV1(input_size=use_col - 1, length_prediction=delay, num_layers=experiment['num_layers'],
                           bidirectional=experiment['bidirectional'], dropout=experiment['dropout'],
                           hidden_size=experiment['hidden_size'])
    elif 'Seq2Seq' in experiment['model']:
        from module.models.seq2seq import Seq2Seq
        from module.models.self_attention import BahdanauAttention

        m = Seq2Seq(input_dim=use_col - 1, enc_hidden_dim=experiment['hidden_size'],
                    dec_hidden_dim=experiment['hidden_size'], enc_num_layers=experiment['num_layers'],
                    dec_num_layers=experiment['num_layers'], dec_bidirectional=experiment['bidirectional'],
                    dropout=experiment['dropout'], attention=BahdanauAttention(enc_hidden_dim=experiment['hidden_size'],
                                                                               dec_hidden_dim=experiment[
                                                                                   'hidden_size']))
    elif 'Transformer' in experiment['model']:
        from module.models.transformer import Transformer

        m = Transformer(input_dim=use_col - 1, out_dim=1, n_head=7, dropout=experiment['dropout'])
    else:
        raise ValueError('Model param:', {experiment['model']})
    print('model: {}'.format(experiment['model']))
    optimiser = optim.Adam(m.parameters(), lr=0.02)
    train_loss_y = []
    val_loss_y = {'val: ': [], 'test: ': []}
    trainer = BasicTrainer(loss_func=loss_fn, cuda=False)
    best_val = 0.
    for i in range(n_epoch):
        common_loss, evaluation_values = trainer.train_per_epoch(train_dataset, m, optimiser, evaluation=evaluation,
                                                                 info=f'epoch: {i + 1}/{n_epoch}',
                                                                 is_seq2seq=True if 'Seq2Seq' in experiment['model']
                                                                                    or
                                                                                    'Transformer' in experiment['model']
                                                                 else False)
        train_loss_y.append(common_loss)
        tb_logger.log_value('train： mae_loss', common_loss, i + 1)
        for key in evaluation_values.keys():
            tb_logger.log_value(f'train: {key}', evaluation_values[key], i + 1)
        for sub, dataset in (('val: ', val_dataset), ('test: ', test_dataset)):
            common_loss, evaluation_values = trainer.validation(dataset, m, evaluation=evaluation,
                                                                is_transformer=True if 'Transformer' in experiment['model']
                                                                else False)
            if 'val' in sub and (i == 0 or common_loss < best_val):
                best_val = common_loss
                torch.save({
                    'epoch': i + 1,
                    'state_dict': m.state_dict(),
                    'optimizer': optimiser.state_dict()
                }, os.path.join(directory, 'best.pth.tar'))
                print(f'model state_dict has been saved in {directory}/best.pth.tar')
            val_loss_y[sub].append(common_loss)
            tb_logger.log_value(f'{sub}mae_loss', common_loss, i + 1)
            for key in evaluation_values.keys():
                tb_logger.log_value(''.join([sub, key]), evaluation_values[key], i + 1)
            print(dict2str(evaluation_values))
        print(f'{i + 1}' + '**' * 50 + '\n')
    print(f'log dir:{directory}')
    torch.save({
        'epoch': n_epoch,
        'state_dict': m.state_dict(),
        'optimizer': optimiser.state_dict()
    }, os.path.join(directory, 'last.pth.tar'))
    print(f'model state_dict has been saved in {directory}/last.pth.tar')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(dpi=600)
    plt.title('{}_{}_loss'.format(experiment['model'], experiment['loss']))
    plt.plot(list(range(n_epoch)), train_loss_y, label='train mae loss', color='red')
    plt.plot(list(range(n_epoch)), val_loss_y['val: '], label='val mae loss', color='green')
    plt.plot(list(range(n_epoch)), val_loss_y['test: '], label='test mae loss', color='blue')
    plt.legend()
    plt.savefig(os.path.join(directory, '{}_loss.tif'.format(experiment['model'])), bbox_inches='tight')
    plt.show()
