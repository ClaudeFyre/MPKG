from torch import nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, opt, adj):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        m = torch.mm(x, self.weight)
        m = torch.spmm(self.adj, m)
        return m

class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.bn1 = None
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x, opt):
        self.bn1 = nn.BatchNorm1d(opt['hidden_dim'])
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x


class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.bn1 = None
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x, opt):
        self.bn1 = nn.BatchNorm1d(opt['hidden_dim'])
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x
