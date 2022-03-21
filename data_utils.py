import os
import numpy as np
import scipy.linalg as linalg
import torch

from utils import numpy2torch, torch2numpy

def pad(seqs, max_len, n_nodes):
    '''
    pad sequences with zeros so that all sequences 
    have same number (=max_len) of time samples
    '''
    seqs_padded = []
    for seq in seqs:
        length = len(seq)
        seq = np.concatenate([seq, np.zeros([max_len-length, n_nodes])], axis=0)
        seqs_padded.append({'len': length, 'x': numpy2torch(seq)})
    return seqs_padded


def unpad(batch, lengths):
    '''
    reverse function of pad.
    reverts all sequences to their original length
    '''
    unpad_batch = [batch[i][:lengths[i]] for i in range(len(batch))]
    return unpad_batch 


def augment_rbf_coefs(seqs_padded, PHI, lhs):
    '''
    computes RBF coefficients corresponding to raw observations
    by solving regularized least square.
    augments the RBF coefficients to the input dictionary data 
    '''
    for i, data in enumerate(seqs_padded):
        rhs = torch.matmul(PHI.t(), data['x'].t())
        LAMBDA, _, _, _ = linalg.lstsq(torch2numpy(lhs), torch2numpy(rhs))
        LAMBDA = numpy2torch(LAMBDA).t()
        data['lambda'] = LAMBDA
        seqs_padded[i] = data
    return seqs_padded


def load_dataset(data_dir, dataset):
    '''
    loads a given dataset from the data directory 
    and splits into train-val-test sets
    '''
    if dataset == 'convdiff':
        path = os.path.join(data_dir, 'irregular_varicoef_diff_conv_eqn_4nn_42_250sample/')
        sim = []
        for i in range(1000):
            data = np.load(path+'{:03d}_simdata.npz'.format(i))
            sim.append(data['frames'])
        sim = np.array(sim)

        p = np.load(path+'node_meta.npy')
        n_nodes = len(p)

        # remove the duplicate and very close data sites to avoid nonsingular RBF matrix
        remove = [16, 18, 19, 27, 32, 35, 45, 64, 79, 90, 95, 96, 118, 146, 191, 210, 216, 217]
        idxs = [k for k in range(n_nodes) if k not in remove]      

        sim = sim[:, :, idxs]
        p = p[idxs]
        
        test_set = sim[850:]
        train_set = sim[:700]
        val_set = sim[700:850]

    if dataset == 'sst':
        path = os.path.join(data_dir, 'sst/sst-daily_4nn_42_250sample/')
        sst = []
        for i in range(24):
            data = np.load(path+'{:03d}_simdata.npz'.format(i))
            sst.append(data['frames'])
        sst = np.array(sst)

        p = np.load(path+'node_meta.npy')
        n_nodes = len(p)

        # remove the duplicate and very close data sites to avoid nonsingular RBF matrix
        remove = [66, 190, 192, 211]
        idxs = [k for k in range(n_nodes) if k not in remove]

        sst = sst[:, :, idxs]
        p = p[idxs]
        p = (p -  p.min(axis=0)) / (p.max(axis=0) - p.min(axis=0))

        test_set = sst[12:]
        val_set = test_set
        train_set = sst[:12]

    if dataset == 'noaa_pt':
        path = os.path.join(data_dir, 'noaa_pt_states_withloc/')
        temp = []
        for i in range(153):
            data = np.load(path+'{:03d}_seq.npz'.format(i))
            temp.append(data['vals'][:, :, 1])

        test_set = temp[124:]
        train_set = temp[:98]
        val_set = temp[98:124]

        p = np.load(path+'node_meta.npy')[:, :2]
        p = (p - p.mean(axis=0)) / p.std(axis=0)

    if dataset == 'noaa_ec':
        path = os.path.join(data_dir, 'noaa_ec_states_test_2_withloc/')
        temp = []
        for i in range(280):
            data = np.load(path+'{:03d}_seq.npz'.format(i))
            temp.append(data['vals'][:, :, 1])

        test_set = temp[219:]
        train_set = temp[:164]
        val_set = temp[164:219]
        
        p = np.load(path+'node_meta.npy')[:, :2]
        p = (p - p.mean(axis=0)) / p.std(axis=0)

    return train_set, val_set, test_set, p
