
import torch
import math
import torch.nn as nn
from torch.nn import init
import numpy as np

import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    AvgPool1d,
    Sigmoid,
    Conv1d,
)
from torch_scatter import scatter_mean

# change the code below to be binary classifiers


class MNIST_Adder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(MNIST_Adder, self).__init__()
        p = 0.3
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_size1, hidden_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(inplace=True)
        )
        self.adder = InvLinear(hidden_size2, hidden_size2, reduction='sum', bias=True)
        self.output_layer = nn.Sequential(nn.ReLU(inplace=True),
                                          nn.Linear(hidden_size2, hidden_size2),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p),
                                          nn.Linear(hidden_size2, hidden_size2),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p),
                                          nn.Linear(hidden_size2, hidden_size2),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p),
                                          nn.Linear(hidden_size2, 1),
                                          nn.Sigmoid())

    def forward(self, X, mask=None):
        N, features, arguments  = X.shape
        h = self.feature_extractor(X.reshape(N, arguments, features))
        h = self.adder(h, mask=mask)
        y = self.output_layer(h)
        return y
    



## is this deep?
    #and if not can we make it deeper?

class DeepSets(torch.nn.Module):
    def __init__(self, input_size, max_len_brief , hidden1, hidden2, hidden3, classify1, p = 0.2):
        super(DeepSets, self).__init__()

        self.narguments = max_len_brief
        p = 0.3
        self.phi = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden2, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden3, hidden3),
            nn.ReLU(inplace=True)
        )

        self.rho = Seq(
            Lin(hidden3, hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            Lin(hidden3, classify1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(classify1, classify1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            Lin(classify1, 1),
            Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        out = self.phi(x)
        # indices = torch.LongTensor(np.zeros(self.narguments)).to(self.device)
        #out = scatter_mean(out, indices, dim=-1)
        out = torch.sum(out, dim=1, dtype=torch.float32)
        return self.rho(torch.squeeze(out, dim=1))



class DeepSetsCNN(torch.nn.Module):
    def __init__(self, inputs, max_len_brief , hidden1, hidden2, hidden3, classify1):
        super(DeepSetsCNN, self).__init__()

        self.narguments = max_len_brief
        
        self.phi = Seq(
            Conv1d(inputs, hidden1, 1),
            BatchNorm1d(hidden1),
            ReLU(),
            Conv1d(hidden1, hidden2, 1),
            BatchNorm1d(hidden2),
            ReLU(),
            Conv1d(hidden2, hidden3, 1),
            BatchNorm1d(hidden3),
            ReLU(),
        )

        self.rho = Seq(
            Lin(hidden3, classify1),
            BatchNorm1d(classify1),
            ReLU(),
            Lin(classify1, 1),
            Sigmoid(),
        )

    def forward(self, x):
        out = self.phi(x)
        # indices = torch.LongTensor(np.zeros(self.narguments)).to(self.device)
        #out = scatter_mean(out, indices, dim=-1)
        out = torch.mean(out, dim=-1, dtype=torch.float32)
        return self.rho(torch.squeeze(out, dim=1))



class InvLinear(nn.Module):
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """
    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """
        N, M, _ = X.shape
        device = X.device
        y = torch.zeros(N, self.out_features).to(device)
        if mask is None:
            mask = torch.ones(N, M).byte().to(device)

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=1).unsqueeze(1)
            Z = X * mask.unsqueeze(2).float()
            y = (Z.sum(dim=1) @ self.beta)/sizes

        elif self.reduction == 'sum':
            Z = X * mask.unsqueeze(2).float()
            y = Z.sum(dim=1) @ self.beta

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            y = Z.max(dim=1)[0] @ self.beta

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            y = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)
    

class EquivLinear(InvLinear):
    r"""Permutation equivariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """
    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super(EquivLinear, self).__init__(in_features, out_features,
                                          bias=bias, reduction=reduction)

        self.alpha = nn.Parameter(torch.Tensor(self.in_features,
                                               self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        super(EquivLinear, self).reset_parameters()
        if hasattr(self, 'alpha'):
            init.xavier_uniform_(self.alpha)

    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to the output set
        Y = {y_1, ..., y_M} through a permutation equivariant linear transformation
        of the form:
            $y_i = \alpha x_i + \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N sets of same cardinality as in X where each element has dimension
           out_features (tensor with shape (N, M, out_features))
        """
        N, M, _ = X.shape
        device = X.device
        Y = torch.zeros(N, M, self.out_features).to(device)
        if mask is None:
            mask = torch.ones(N, M).byte().to(device)

        Y = torch.zeros(N, M, self.out_features).to(device)
        h_inv = super(EquivLinear, self).forward(X, mask=mask)
        Y[mask] = (X @ self.alpha + h_inv.unsqueeze(1))[mask]

        return Y
