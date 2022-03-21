import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    Module definition of a multi-layer perceptron
    '''
    def __init__(self, in_dim, out_dim, hid_dims=[64, 32], activation=nn.ReLU):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dims = hid_dims

        self.layers = [nn.Linear(in_dim, hid_dims[0]), activation()]
        for i in range(1, len(hid_dims)):
            self.layers.extend([nn.Linear(hid_dims[i-1], hid_dims[i]), activation()])
        self.layers.extend([nn.Linear(hid_dims[-1], out_dim)])

        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.sequential(x)


class SpatialTransform(nn.Module):
    '''
    Module definition of the spatial transformation network
    '''
    def __init__(self, in_dim=5, out_dim=16, hid_dims=[64, 32]):
        super(SpatialTransform, self).__init__()

        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.out_dim = out_dim

        self.mlp = MLP(in_dim, out_dim, hid_dims)

    def forward(self, phi, phiInv, args):
        x = torch.cat([phi, args], dim=-1)
        D = self.mlp(x)
        D = D.permute(2, 0, 1)
        scale = torch.abs(phiInv).max()
        phiInv = phiInv.unsqueeze(0)
        phiInv = phiInv.repeat(self.out_dim, 1, 1)
        S = torch.matmul(phiInv/scale, D)
        I = torch.eye(D.shape[-1]).unsqueeze(0)
        S = (S + I/scale).permute(1, 2, 0)
        return S


class NAB(nn.Module):
    '''
    Module definition of the nonlinear aggregation block
    '''
    def __init__(self, in_dim=16, hid_dims=[32]):
        super(NAB, self).__init__()

        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.out_dim = 1

        self.mlp = MLP(in_dim, self.out_dim, hid_dims)

    def forward(self, x):
        return self.mlp(x)


class RFN(nn.Module):
    '''
    Module definition of the recurrent fusion network
    '''
    def __init__(self, in_dim=16, hid_dim=64, out_dim=1, n_layers=1):
        super(RFN, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            outputs, hidden = self.gru(inputs)
        else:
            outputs, hidden = self.gru(inputs, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden


class RBF(nn.Module):
    '''
    Module definition of radial basis functions
    '''
    def __init__(self, cls='ga', eps=None):
        super(RBF, self).__init__()

        self.cls = cls
        if eps is not None:
            self.eps = nn.Parameter(torch.Tensor([eps]))

    def forward(self, x):
        if not hasattr(self, 'eps'):
            return (x ** 4) * torch.log(x + 1e-5)

        if self.cls == 'mq':
            phi = torch.sqrt(x ** 2 + self.eps ** 2)
        elif self.cls == 'ps3':
            phi = (self.eps * x) ** 3
        elif self.cls == 'ps4':
            phi = ((self.eps * x) ** 4) * torch.log(self.eps * x + 1e-5)
        else:
            phi = torch.exp(- ((x / self.eps) ** 2) / 2)
        return phi
