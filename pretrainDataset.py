from torch.utils.data import DataLoader, Dataset
import torch
import os
import pickle
from tqdm import tqdm
from termcolor import cprint


class PretrainDataloader:
    def __init__(self, config):
        super(PretrainDataloader, self).__init__()
        self.item_count = -1
        self.user_count = -1
        self.valid_time = None
        self.test_time = None
        self.train_list = None
        self.valid_list = None
        self.test_list = None
        self.load_data(config)

    def load_data(self, config):
        local_path = './local_data/' + config['dataset']

        if os.path.isfile(local_path + '/train.pkl'):
            cprint("Load data from Pickle", 'red')
            self.train_list = pickle.load(open(local_path + '/train.pkl', 'rb'))
            self.test_list = pickle.load(open(local_path + '/test.pkl', 'rb'))
            self.valid_list = pickle.load(open(local_path + '/valid.pkl', 'rb'))
            info_dic = pickle.load(open(local_path + '/info.pkl', 'rb'))
            self.item_count = info_dic['item_count']
            self.user_count = info_dic['user_count']
        else:
            cprint("Load and process from .txt")

            train_list = self.read_txt(config['data_path'] + config['dataset'] + '/' + config['dataset'] + '.train.inter')
            valid_list = self.read_txt(config['data_path'] + config['dataset'] + '/' + config['dataset'] + '.valid.inter')
            test_list = self.read_txt(config['data_path'] + config['dataset'] + '/' + config['dataset'] + '.test.inter')
            self.item_count = self.item_count + 1
            self.user_count = self.user_count + 1

            self.train_list = train_list
            self.valid_list = valid_list
            self.test_list = test_list
            self.save_data_pkl('./local_data/' + config['dataset'])
        config['item_count'] = self.item_count
        config['user_count'] = self.user_count
        config['n_train_examples'] = len(self.train_list)
        config['n_valid_examples'] = len(self.valid_list)
        config['n_test_examples'] = len(self.test_list)

        if config['use_text_emb']:
            text_emb = pickle.load(open(config['data_path'] + config['dataset'] + '/' + config['dataset'] + '.feat1CLS', 'rb'))
            text_emb = torch.cat([torch.zeros([1, 768]), text_emb], dim=0)
            config['text_emb'] = text_emb
            assert self.item_count == text_emb.size()[0]

    def read_txt(self, file):
        examples = []
        with open(file, 'r') as rf:
            rf.readline()
            for line in tqdm(rf):
                line = line.strip('\n')
                if len(line) < 0:
                    break

                line_list = line.split('\t')

                # item id都需要+1， 为padding token留位置
                user_id = int(line_list[0])
                item_seq = line_list[1].split(' ')
                item_seq = [int(item)+1 for item in item_seq]
                target_item = int(line_list[2]) + 1
                examples.append([user_id, item_seq, target_item])

                self.item_count = max(self.item_count, max(item_seq), target_item)
                self.user_count = max(self.user_count, user_id)
        return examples

    def generate_dataloader(self, config):
        train_dataset = PretrainDataset(self.train_list, 'train', config)
        valid_dataset = PretrainDataset(self.valid_list, 'valid', config)
        test_dataset = PretrainDataset(self.test_list, 'test', config)

        train_dataloader = DataLoader(train_dataset, batch_size=config['train_batch_size'], collate_fn=train_dataset.collate_fn, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config['test_batch_size'], collate_fn=valid_dataset.collate_fn, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], collate_fn=test_dataset.collate_fn, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader


    def save_data_pkl(self, root_path):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        cprint("Save Pickled data", 'red')
        pickle.dump(self.train_list, open(root_path + '/train.pkl', 'wb'))
        pickle.dump(self.test_list, open(root_path + '/test.pkl', 'wb'))
        pickle.dump(self.valid_list, open(root_path + '/valid.pkl', 'wb'))
        pickle.dump({
            "item_count": self.item_count,
            "user_count": self.user_count,
        }, open(root_path + '/info.pkl', 'wb'))



class PretrainDataset(Dataset):
    def __init__(self, example_list, mode, config):
        """
        """
        super(PretrainDataset, self).__init__()
        self.example_list = example_list
        self.mode = mode
        self.item_count = config['item_count']
        self.pad = 0
        self.length = len(self.example_list)
        self.seq_max_len = config['max_seq_len']
        self.pad_mode = config['pad_mode']
        self.neg_count = config['neg_count'] if mode == 'train' else config['test_neg_count']
        self.use_text_emb = config['use_text_emb']
        self.config = config
        if self.mode == 'valid' and type(self.neg_count) is int and self.config['stage'] == 'pretrain':
            self.neg_sample = torch.randint(low=0, high=self.item_count - 1, size=(len(self.example_list), self.neg_count)).tolist()
            assert len(self.neg_sample) == len(self.example_list)
        else:
            self.neg_sample = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user_id = self.example_list[index][0]
        item_seq = self.example_list[index][1][-self.seq_max_len:]
        label = self.example_list[index][2]
        if self.neg_sample is not None:
            negs = self.neg_sample[index][: self.neg_count]
        else:
            negs = [0]
        return [0] * (self.seq_max_len - len(item_seq)) + item_seq, \
               user_id, \
               label, \
               negs, \
               len(item_seq)

    def collate_fn(self, batch):
        item_seqs = []
        users_id = []
        labels = []
        negs = []
        lengths = []
        for example in batch:
            item_seqs.append(example[0])
            users_id.append(example[1])
            labels.append(example[2])
            negs.append(example[3])
            lengths.append(example[4])

        return {
            'item_seqs': torch.tensor(item_seqs),
            'users': torch.tensor(users_id),
            'labels': torch.tensor(labels),
            'neg_items': torch.tensor(negs),
            'lengths': torch.tensor(lengths),
        }
