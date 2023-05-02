import torch
import torch.optim as optim
from time import localtime
from time import time
from tqdm import tqdm
import os
from termcolor import cprint
from metrics import cal_recall, cal_ndcg
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class PretrainTrainer(object):
    def __init__(self, model, config, train_dataloader, valid_dataloader, test_dataloader):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.losses_types = ['Total Loss']
        self.device = config['device']

        self.optimizer = None
        self.scheduler = None
        self.set_optimizer(self.model, config)

        self.cur_best = 0
        self.stopping_step = 0
        self.should_stop = False
        self.best_recall, self.best_ndcg = None, None

        if self.config['load_model_path'] == "" or self.config['stage'] != 'train':
            local_time = localtime()
            self.saved_path = "./saved_model/{}-{}-{}-{}/".format(local_time.tm_mday,
                                                                  local_time.tm_hour,
                                                                  local_time.tm_min,
                                                                  local_time.tm_sec)
            if not os.path.exists(self.saved_path):
                os.makedirs(self.saved_path)
        else:
            self.saved_path = '/'.join(self.config['load_model_path'].split('/')[:-1]) + '/'

    def set_optimizer(self, model, config):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config['regs'],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=1e-8)

    def train(self):

        for epoch in range(self.config['start_epoch'], self.config['epoch']):
            self.model.epoch_step()
            self.train_epoch(epoch)
            self.valid(epoch)
            if self.should_stop:
                break
        if self.config['stage'] == 'pretrain':
            print("Pretrain Finished!!!")
            exit(0)

        cprint('### Test Model ###', 'red')
        self.model.load_state_dict(torch.load(self.saved_path + 'best_model.pkl'))
        return self.test()

    def train_epoch(self, epoch):
        self.model.train()
        t1 = time()
        total_losses = [0] * len(self.losses_types)
        for batch_idx, data in enumerate(tqdm(self.train_dataloader)):

            self.trans_device(data)
            self.model.batch_step()
            losses = self.model.calculate_loss(data)

            self.optimizer.zero_grad()
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            # self.scheduler.step()

            for i in range(len(losses)):
                total_losses[i] += losses[i].cpu().item()

        loss_str = 'Epoch:{} \tTime:{} \t'.format(epoch, time() - t1)
        for i in range(len(self.losses_types)):
            loss_str += str(self.losses_types[i]) + ":" + str(total_losses[i] / len(self.train_dataloader))[:6] + "\t"
        print(loss_str)

    @torch.no_grad()
    def valid(self, epoch=0, batch=0):
        self.model.eval()
        t0 = time()
        label = []
        predict_score = []
        for batch_idx, data in enumerate(self.valid_dataloader):
            if batch != 0 and batch_idx > int(len(self.valid_dataloader) / 10):
                break
            self.trans_device(data)
            if self.config['stage'] == 'pretrain':
                batch_score, batch_label = self.model.predict(data)
            else:
                batch_score, batch_label = self.model.full_sort_predict(data)
            predict_score.append(batch_score.cpu())
            label.append(batch_label.cpu())  # 进行负采样
        recall = cal_recall(label, predict_score, self.config['ks'])
        ndcg = cal_ndcg(label, predict_score, self.config['ks'])
        t1 = time()

        result = {
            'recall': recall,
            'ndcg': ndcg,
        }
        perf_str = 'Valid Epoch-Batch %d-%d [%.1fs]: recall=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (
            epoch, batch, t1 - t0, recall[1], recall[-2], ndcg[1], ndcg[-2])
        cprint(perf_str, 'red')

        if batch == 0:
            self.early_stop(result, flag_step=self.config['flag_step'], epoch=epoch)

        return result

    @torch.no_grad()
    def test(self):
        self.model.eval()
        t0 = time()
        label = []
        predict_score = []
        for batch_idx, data in enumerate(self.test_dataloader):
            self.trans_device(data)
            if self.config['stage'] == 'pretrain':
                batch_score, batch_label = self.model.predict(data)
            else:
                batch_score, batch_label = self.model.full_sort_predict(data)
            predict_score.append(batch_score.cpu())
            label.append(batch_label.cpu())
        recall = cal_recall(label, predict_score, self.config['ks'])
        ndcg = cal_ndcg(label, predict_score, self.config['ks'])
        t1 = time()

        result = {
            'recall': recall,
            'ndcg': ndcg,
        }
        perf_str = 'Test Result [%.1fs]: recall=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (
            t1 - t0, recall[1], recall[-2], ndcg[1], ndcg[-2])
        cprint(perf_str, 'red')
        return result

    def early_stop(self, result, metric='recall', flag_step=5, epoch=0):
        if self.config['stage'] == 'pretrain':
            if epoch % self.config['save_step'] == 0:
                torch.save(self.model.state_dict(), self.saved_path + f'pretrain-{epoch}.pkl')
                cprint("Saving the weights in path: " + self.saved_path + f'pretrain-{epoch}.pkl', 'green')
            return

        if result[metric][1] > self.cur_best:
            self.cur_best = result[metric][1]
            self.stopping_step = 0
            self.should_stop = False
            self.best_recall = result['recall']
            self.best_ndcg = result['ndcg']
            if self.config['save_flag'] == 1:
                torch.save(self.model.state_dict(), self.saved_path + 'best_model.pkl')
                cprint("Saving the weights in path: " + self.saved_path + 'best_model.pkl', 'green')
        elif self.stopping_step + 1 <= flag_step:
            self.stopping_step = self.stopping_step + 1
            self.should_stop = False
            torch.save(self.model.state_dict(), self.saved_path + f'step_{self.stopping_step}.pkl')
            cprint(f'cur_step:{self.stopping_step}', 'green')
        else:
            self.should_stop = True

    def trans_device(self, data):
        for item in data.keys():
            if type(data[item]) == torch.Tensor:
                data[item] = data[item].to(self.device)

