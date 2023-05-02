"""
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
"""

import torch
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


set_seed(2023)
import sys

sys.path.append('..')

from parser import get_config
from pretrainTrainer import PretrainTrainer
from pretrainDataset import PretrainDataloader
from miracle import Miracle
import warnings

warnings.filterwarnings('ignore')


def print_config(config):
    for item in config.keys():
        print(str(item) + ':' + str(config[item]))


def train_one_model(dataset=None, load_model_epoch=None):
    config = get_config()
    if dataset is not None:
        config['dataset'] = dataset
    if load_model_epoch is not None:
        config['load_model_path'] = config['load_model_path'][:-5] + str(load_model_epoch)+'.pkl'
        print("Load model path:", config['load_model_path'])
    config['device'] = torch.device('cuda:' + str(config['gpu_id'])) if config['cuda'] else torch.device('cpu')
    dataloader = PretrainDataloader(config)
    train_dataloader, valid_dataloader, test_dataloader = dataloader.generate_dataloader(config)
    print_config(config)
    # 初始化模型
    model = Miracle(config)
    model = model.to(config['device'])


    if config['stage'] != 'pretrain' and not config['load_model_path']:
        raise NotImplementedError   # downstream stage needs loading pretrained model

    # 加载旧模型
    if config['load_model_path']:
        model.load_state_dict(torch.load(config['load_model_path'], map_location=config['device']), strict=False)
        if config['stage'] != 'pretrain' and config['freeze']:
            model.downstream_freeze_parameter()

    trainer = PretrainTrainer(model, config, train_dataloader, valid_dataloader, test_dataloader)
    # trainer.analysis_interest_similarity()
    # exit()
    trainer.valid(0)
    return trainer.train()


def p_test(times):
    datasets = ['Scientific', 'Pantry', 'Instruments', 'OR', 'Arts', 'Office']
    # datasets = ['Arts', 'Office']

    for dataset in datasets:
        for i in range(times):
            result = train_one_model(dataset)
            hit10, hit50, ndcg10, ndcg50 = result['recall'][1], result['recall'][3], result['ndcg'][1], result['ndcg'][3]
            with open(f'./p_test_result.txt', 'a+') as wf:
                wf.write(f"{dataset}\t{hit10}\t{hit50}\t{ndcg10}\t{ndcg50}\n")




if __name__ == '__main__':
    # for dataset in ['Scientific', 'Pantry']:
    # for dataset in ['Scientific', 'Pantry', 'Instruments', 'OR', 'Arts', 'Office', 'Games']:
    #     train_one_model(dataset)
    #     exit()
    # train_one_model('Office')
    train_one_model()

    # for epoch in range(2, 5):
    #     train_one_model('Scientific', epoch)
    # train_one_model()
    # p_test(2)