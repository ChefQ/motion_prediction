
import torch
import math
import torch.nn as nn
from torch.nn import init
import numpy as np



import torch.nn.init
import math

from utils import linear_block

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
#from torch_scatter import scatter_mean

# change the code below to be binary classifiers


class MNIST_Adder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(MNIST_Adder, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(inplace=True)
        )
        self.adder = InvLinear(hidden_size2, hidden_size2, reduction='sum', bias=True)
        self.output_layer = nn.Sequential(nn.ReLU(inplace=True),
                                          nn.Linear(30, 1),
                                          nn.Sigmoid())

    def forward(self, X, mask=None):
        N, S, C, D, _ = X.shape
        h = self.feature_extractor(X.reshape(N, S, C*D*D))
        h = self.adder(h, mask=mask)
        y = self.output_layer(h)
        return y
    
 

## is this deep?
    #and if not can we make it deeper?

class DeepSets(torch.nn.Module):
    def __init__(self, inputs, max_len_brief , hidden1, hidden2, hidden3, classify1):
        super(DeepSets, self).__init__()

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


class MultiSetTransformer(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=2, remove_diag=False, ln=False, equi=False, 
            weight_sharing='none', dropout=0.1, decoder_layers=0, pool='pma', merge='concat'):
        super(MultiSetTransformer, self).__init__()
        if equi:
            input_size = 1
        self.input_size = input_size
        self.proj = None if input_size == latent_size else nn.Linear(input_size, latent_size) 
        self.enc = EncoderStack(*[CSAB(latent_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, 
                equi=equi, weight_sharing=weight_sharing, dropout=dropout, merge='concat') for i in range(num_blocks)])
        self.pool_method = pool
        if self.pool_method == "pma":
            self.pool_x = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
            self.pool_y = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
        self.dec = self._make_decoder(latent_size, hidden_size, output_size, decoder_layers)
        self.remove_diag = remove_diag
        self.equi=equi

    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers):
        if n_layers == 0:
            return nn.Linear(2*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, X, Y, masks=None):
        ZX, ZY = X, Y
        if self.equi:
            ZX, ZY = ZX.unsqueeze(-1), ZY.unsqueeze(-1)
        if self.proj is not None:

            X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
            Y = Y.reshape(Y.shape[0], Y.shape[2], Y.shape[1])
            ZX = self.proj(ZX)
            ZY = self.proj(ZY)
            
        ZX, ZY = self.enc((ZX, ZY), masks=masks)
            
        if self.equi:
            ZX = ZX.max(dim=2)[0]
            ZY = ZY.max(dim=2)[0]
        
        #backwards compatibility
        if getattr(self, "pool_method", None) is None or self.pool_method == "pma":
            ZX = self.pool_x(ZX)
            ZY = self.pool_y(ZY)
        elif self.pool_method == "max":
            ZX = torch.max(ZX, dim=1)
            ZY = torch.max(ZY, dim=1)
        elif self.pool_method == "mean":
            ZX = torch.mean(ZX, dim=1)
            ZY = torch.mean(ZY, dim=1)

        out = self.dec(torch.cat([ZX, ZY], dim=-1))
        return out.squeeze(-1)



class PMA(nn.Module):
    def __init__(self, latent_size, hidden_size, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, latent_size))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(latent_size, latent_size, hidden_size, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)



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



class EncoderStack(nn.Sequential):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


class CSAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, remove_diag=False, nn_attn=False, residual='base', weight_sharing='none', merge='concat', ln=False, lambda0=0.5, **kwargs):
        super(CSAB, self).__init__()
        self._init_blocks(input_size, latent_size, hidden_size, num_heads, remove_diag, nn_attn, weight_sharing, ln=ln, merge=merge, **kwargs)
        self.merge = merge
        self.remove_diag = remove_diag

    def _init_blocks(self, input_size, latent_size, hidden_size, num_heads, remove_diag=False, nn_attn=False, weight_sharing='none', ln=False, merge='concat', **kwargs):
        if weight_sharing == 'sym':
            MAB_cross = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, ln=ln, **kwargs)
            MAB_self = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, ln=ln, **kwargs)
            self.MAB_XX = MAB_self
            self.MAB_YY = MAB_self
            self.MAB_XY = MAB_cross
            self.MAB_YX = MAB_cross
            if merge == 'concat':
                fc = nn.Linear(2*latent_size, latent_size)
                self.fc_X = fc
                self.fc_Y = fc
            if ln:
                lns = nn.LayerNorm(latent_size)
                self.ln_x = lns
                self.ln_y = lns
        else:
            if merge == 'concat':
                self.fc_X = nn.Linear(2*latent_size, latent_size)
                self.fc_Y = nn.Linear(2*latent_size, latent_size)
            if ln:
                self.ln_x = nn.LayerNorm(latent_size)
                self.ln_y = nn.LayerNorm(latent_size)

            if weight_sharing == 'none':
                self.MAB_XX = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
                self.MAB_YY = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
                self.MAB_XY = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
                self.MAB_YX = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            elif weight_sharing == 'cross':
                self.MAB_XX = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
                self.MAB_YY = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
                MAB_cross = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
                self.MAB_XY = MAB_cross
                self.MAB_YX = MAB_cross
            else:
                raise NotImplementedError("weight sharing must be none, cross or sym")

    def _get_masks(self, N, M, masks):
        if self.remove_diag:
            diag_xx = (1 - torch.eye(N)).unsqueeze(0)
            diag_yy = (1 - torch.eye(M)).unsqueeze(0)
            if use_cuda:
                diag_xx = diag_xx.cuda()
                diag_yy = diag_yy.cuda()
            if masks is not None:
                mask_xx, mask_xy, mask_yx, mask_yy = masks
                mask_xx = mask_xx * diag_xx
                mask_yy = mask_yy * diag_yy
            else:
                mask_xx, mask_yy = diag_xx, diag_yy
                mask_xy, mask_yx = None, None
        else:
            if masks is not None:
                mask_xx, mask_xy, mask_yx, mask_yy = masks 
            else: 
                mask_xx, mask_xy, mask_yx, mask_yy = None,None,None,None
        return mask_xx, mask_xy, mask_yx, mask_yy

    def forward(self, inputs, masks=None, neighbours=None):
        X, Y = inputs
        mask_xx, mask_xy, mask_yx, mask_yy = self._get_masks(X.size(1), Y.size(1), masks)
        XX = self.MAB_XX(X, X, mask=mask_xx)
        XY = self.MAB_XY(X, Y, mask=mask_xy)
        YX = self.MAB_YX(Y, X, mask=mask_yx)
        YY = self.MAB_YY(Y, Y, mask=mask_yy)
        if self.merge == "concat":
            X_merge = self.fc_X(torch.cat([XX, XY], dim=-1))
            Y_merge = self.fc_Y(torch.cat([YY, YX], dim=-1))
        else:
            X_merge = XX + XY
            Y_merge = YX + YY
        X_out = X + X_merge
        Y_out = Y + Y_merge
        X_out = X_out if getattr(self, 'ln_x', None) is None else self.ln_x(X_out)
        Y_out = Y_out if getattr(self, 'ln_y', None) is None else self.ln_y(Y_out)
        return (X_out, Y_out)


class MHA(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, bias=None, equi=False, nn_attn=False):
        super(MHA, self).__init__()
        if bias is None:
            bias = not equi
        self.latent_size = dim_V
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_Q, dim_V, bias=bias)
        self.w_k = nn.Linear(dim_K, dim_V, bias=bias)
        self.w_v = nn.Linear(dim_K, dim_V, bias=bias)
        self.w_o = nn.Linear(dim_V, dim_V, bias=bias)
        self.equi = equi
        self.nn_attn = nn_attn

    def _mha(self, Q, K, mask=None):
        Q_ = self.w_q(Q)
        K_, V_ = self.w_k(K), self.w_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.stack(Q_.split(dim_split, 2), 0)
        K_ = torch.stack(K_.split(dim_split, 2), 0)
        V_ = torch.stack(V_.split(dim_split, 2), 0)

        E = Q_.matmul(K_.transpose(2,3))/math.sqrt(self.latent_size)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_)).split(1, 0), 3).squeeze(0))
        return O
    
    def _equi_mha(self, Q, K, mask=None):
        # band-aid fix for backwards compat:
        d = self.latent_size if getattr(self, 'latent_size', None) is not None else self.dim_V

        Q = self.w_q(Q)
        K, V = self.w_k(K), self.w_v(K)

        dim_split = d // self.num_heads
        Q_ = torch.stack(Q.split(dim_split, 3), 0)
        K_ = torch.stack(K.split(dim_split, 3), 0)
        V_ = torch.stack(V.split(dim_split, 3), 0)

        E = Q_.transpose(2,3).matmul(K_.transpose(2,3).transpose(3,4)).sum(dim=2) / math.sqrt(d)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_.view(*V_.size()[:-2], -1)).view(*Q_.size())).split(1, 0), 4).squeeze(0))
        return O

    def forward(self, *args, **kwargs):
        if getattr(self, 'equi', False):
            return self._equi_mha(*args, **kwargs)
        else:
            return self._mha(*args, **kwargs)



class MAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, attn_size=None, ln=False, rezero=False, equi=False, nn_attn=False, dropout=0.1):
        super(MAB, self).__init__()
        attn_size = attn_size if attn_size is not None else input_size
        self.attn = MHA(input_size, attn_size, latent_size, num_heads, equi=equi, nn_attn=nn_attn)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)
        if rezero:
            self.alpha0 = nn.Parameter(torch.tensor(0.))
            self.alpha1 = nn.Parameter(torch.tensor(0.))
        else:
            self.alpha0 = 1
            self.alpha1 = 1

    def forward(self, Q, K, **kwargs):
        X = Q + getattr(self, 'alpha0', 1) * self.attn(Q, K, **kwargs)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln0', None) is None else self.ln0(X)
        X = X + getattr(self, 'alpha1', 1) * self.fc(X)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln1', None) is None else self.ln1(X)
        return X
