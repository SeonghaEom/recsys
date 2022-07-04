import lib
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.candid2idx_df = pd.read_csv('./notebooks/candid2idx_df.csv')
        self.candidate = torch.FloatTensor(self.candid2idx_df['idx'].to_list())
        # self.predicted = pd.DataFrame(columns=self.candid2idx_df.columns)
        self.predicted = []

    def select_from_candidate(self, logit, sid):

        print(logit.shape)
        sorted, indices = torch.sort(logit, descending=True)
        indices = indices.cpu()
        # print(indices[:,:100].get_device(), candidate.get_device())
        print(torch.count_nonzero(torch.isin(indices[:,:3].cpu(), self.candidate)))

        top100 = []
        for i, row in enumerate(indices): # B, 4990?
            mask = torch.isin(row[:100], self.candidate)
            incand = torch.masked_select(row[:100], mask)
            assert incand.shape[0] >= 100, incand.shape
            top100.append(incand[:100])

        top100 = torch.Tensor(top100)
        print(torch.count_nonzero(torch.isin(top100, self.candidate)))
        top100 = indices[:,:100].tolist()
        for i, row in enumerate(top100): # (B, 100)
            for j, idx in enumerate(row): # 
                # print( self.candid2idx_df.loc[self.candid2idx_df['idx'] == idx]['item_id'])
                sid_iid_rank = (sid, self.candid2idx_df.loc[self.candid2idx_df['idx'] == idx]['item_id'], j)
                self.predicted.append(sid_iid_rank)
        

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        # dataloader = lib.DataLoader(eval_data, batch_size)
        dataloader = eval_data
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, p_target, mask, sid, flag) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            #for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)
                p_target = p_target.to(self.device)
                logit, hidden = self.model(input, hidden, flag)
                # print("logit shape ", logit.shape) # (B, num of items 23691)
                if flag:
                    
                    # self.select_from_candidate(logit, sid)

                    
                    p_logit_sampled = logit[:, p_target.view(-1)]
                    p_loss = self.loss_func(p_logit_sampled)
                    recall, mrr = lib.evaluate(logit, p_target, k=self.topk)

                    # torch.Tensor.item() to get a Python number from a tensor containing a single value
                    losses.append(p_loss.item())
                    recalls.append(recall)
                    mrrs.append(mrr.item())
                else:
                    logit_sampled = logit[:, target.view(-1)]
                    loss = self.loss_func(logit_sampled)
                    losses.append(loss.item())

        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)
        # pd.DataFrame.from_records(self.predicted,  columns=['session_id', 'item_id', 'rank']).to_csv("./predicted.csv")

        return mean_losses, mean_recall, mean_mrr