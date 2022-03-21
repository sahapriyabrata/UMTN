import numpy as np
import scipy.linalg as linalg
import torch

from utils import torch2numpy, numpy2torch
from data_utils import unpad

def sample_independent(spatial, PHI, invPHI, ARGS, batch_size):
    '''
    computes the spatial features, independent of the observed data.
    requires only the data sites and the RBF matrix
    '''
    S = spatial(PHI.unsqueeze(-1), invPHI, ARGS)
    S = S.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return S


def sample_dependent(nab, rfn, seqs, S, PHI, lhs, 
                     batch_size, n_nodes, in_len, max_len, n_levels, n_spatial_fts, 
                     train_sample_prob=None):
    '''
    forward passes through the model.
    returns the true observations and the predictions 
    '''
    X = [data['x'] for data in seqs]
    X = torch.stack(X)
    LAMBDA = [data['lambda'] for data in seqs]
    LAMBDA = torch.stack(LAMBDA)
    C_ = LAMBDA[:, :in_len].reshape(batch_size, 1, -1, n_nodes)
    Cs = []
    for _ in range(n_levels):
        C = torch.matmul(C_.repeat(1, n_nodes, 1, 1), S)
        Cs.append(C)
        C_ = nab(C)
        C_ = C_.permute(0, 3, 2, 1)
    Cs = torch.cat(Cs, dim=-1)
    Cs = Cs.permute(3, 0, 1, 2)
    PHI_expd = PHI.reshape(1, 1, *PHI.shape)
    PHI_expd = PHI_expd.repeat(n_levels*n_spatial_fts, batch_size, 1, 1)
    V = torch.matmul(PHI_expd, Cs)
    V = V.permute(1, 2, 3, 0)
    U = X[:, :in_len].transpose(-1, -2).unsqueeze(-1)
    V = torch.cat([V, U], dim=-1)
    V = V.reshape(-1, *V.shape[-2:])
    U_, hidden = rfn(V)
    U_ = U_.reshape(batch_size, n_nodes, -1)
    pred = U_.transpose(-1, -2)
    for t in range(in_len, max_len-1):
        random_float = np.random.uniform(0.0, 1.0)
        if (train_sample_prob is not None) and (random_float < train_sample_prob):
            nextX = X[:, t]
            nextLAMBDA = LAMBDA[:, t]
        else:
            nextX = pred[:, -1].detach()
            rhs = torch.matmul(PHI.t(), nextX.t())
            nextLAMBDA, _, _, _ = linalg.lstsq(torch2numpy(lhs), torch2numpy(rhs))
            nextLAMBDA = numpy2torch(nextLAMBDA).t()
        C_ = nextLAMBDA.reshape(batch_size, 1, -1, n_nodes)
        Cs = []
        for _ in range(n_levels):
            C = torch.matmul(C_.repeat(1, n_nodes, 1, 1), S)
            Cs.append(C)
            C_ = nab(C)
            C_ = C_.permute(0, 3, 2, 1)
        Cs = torch.cat(Cs, dim=-1)
        Cs = Cs.permute(3, 0, 1, 2)
        V = torch.matmul(PHI_expd, Cs)
        V = V.permute(1, 2, 3, 0)
        U = nextX.unsqueeze(-1).unsqueeze(-1)
        V = torch.cat([V, U], dim=-1)
        V = V.reshape(-1, *V.shape[-2:])
        U_, hidden = rfn(V, hidden)
        U_ = U_.reshape(batch_size, 1, n_nodes)
        pred = torch.cat([pred, U_], dim=1)
    target = X[:, 1:]
    lengths = [data['len']-1  for data in seqs]
    target = unpad(target, lengths)
    pred = unpad(pred, lengths)
    return target, pred
