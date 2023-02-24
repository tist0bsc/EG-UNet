import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, g_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features        
        self.weight=nn.Linear(in_features,out_features,bias=g_bias)


    def forward(self, input, adj):
        b,c,h,w=input.size()
        input=input.view(b,c,-1)    
        input=input.permute(0,2,1)
        support = self.weight(input)
        output = torch.bmm(adj, support)
        output = output.permute(0,2,1)
        output = output.view(b,-1,h,w)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'