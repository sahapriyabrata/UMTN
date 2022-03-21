import csv
import numpy as np
import torch

class Config:
   def __init__(self, config_file):
       self.load_config(config_file)
       
   def load_config(self, config_file):
       '''
       loads the input arguments from the config file
       '''
       with open(config_file) as f:
           contents = csv.reader(f, delimiter=' ')
           for row in contents:
               k, t, v = row
               if t == 'float':
                   v = float(v)
               if t == 'int':
                   v = int(v)
               if t == 'list':
                   v = v.split(',')
               setattr(self, k, v)


def torch2numpy(q):
    if q.is_cuda:
        q = q.cpu()
    if q.requires_grad:
        q = q.detach()
    return q.numpy()


def numpy2torch(q):
    q = torch.from_numpy(q).float()
    if torch.cuda.is_available():
        q = q.cuda()
    return q


def get_MAEs(gt, pred, mae_list, in_len):
    '''
    computes multi-step mean absolute errors
    '''
    maes = []
    for item in mae_list:
        try:
            item = int(item)
        except:
            continue
        mae = [np.abs(gt[k][in_len-1:in_len+item-1] - pred[k][in_len-1:in_len+item-1]).mean() for k in range(len(gt)) if len(gt[k]) > in_len+item-2]
        maes.append(np.array(mae).mean())
    return maes
