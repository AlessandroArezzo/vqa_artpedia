from __future__ import print_function
import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm

def get_act(act):
    if act == "ReLU":
        act_layer = nn.ReLU
    elif act == "Tanh":
        act_layer = nn.Tanh
    elif act == "Sigmoid":
        act_layer = nn.Sigmoid
    elif act == None:
        act_layer = None
    else:
        print("invalid activation function")
    return act_layer

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act, dropout):
        super(FCNet, self).__init__()
        
        act_layer = get_act(act)
        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(act_layer())
            #layers.append(nn.Dropout(p=dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(act_layer())
        #layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SimpleFC(nn.Module):
    def __init__(self, dims, act, dropout):
        super(SimpleFC, self).__init__()
        act_layer = get_act(act)
        layers = []
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act_layer != None:
            layers.append(act_layer())
        layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        
class GTH(nn.Module):
    def __init__(self, dims,act,dropout):
        super(GTH, self).__init__()
        #in_dim = dims[i]
        #out_dim = dims[i+1]
        self.nonlinear = FCNet(dims,act=act, dropout=dropout)
        self.gate = FCNet(dims,act="Sigmoid", dropout=dropout)
        
    def forward(self, x):
        x_proj = self.nonlinear(x)
        gate = self.gate(x)
        x_proj = x_proj*gate
        return x_proj#self.main(x)#self.nonlinear_out(th_act,sig_act)


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
