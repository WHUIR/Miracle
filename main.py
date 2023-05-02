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


def train_one_model():
    config = get_config()
    config['device'] = torch.device('cuda:' + str(config['gpu_id'])) if config['cuda'] else torch.device('cpu')
    dataloader = PretrainDataloader(config)
    train_dataloader, valid_dataloader, test_dataloader = dataloader.generate_dataloader(config)
    print_config(config)
    model = Miracle(config)
    model = model.to(config['device'])

    if config['stage'] != 'pretrain' and not config['load_model_path']:
        raise NotImplementedError   # downstream stage needs pretrained model

    # load pretrain model
    if config['load_model_path']:
        model.load_state_dict(torch.load(config['load_model_path'], map_location=config['device']), strict=False)
        if config['stage'] != 'pretrain' and config['freeze']:
            model.downstream_freeze_parameter()

    trainer = PretrainTrainer(model, config, train_dataloader, valid_dataloader, test_dataloader)
    trainer.valid(0)
    return trainer.train()


if __name__ == '__main__':
    train_one_model()
