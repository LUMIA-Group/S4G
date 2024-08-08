import math
import torch
from torch.nn import Parameter, LayerNorm, Sequential, GELU, Linear
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from layer.s4 import SSMKernelDPLR


class S4Layer(MessagePassing):
    def __init__(self, params):
        super().__init__(aggr='add', node_dim=0)
        self.params = params

        # SSM parameters
        self.V = Linear(params['hidden_channel'], params['hidden_channel'])
        self.O = Linear(params['hidden_channel'], params['hidden_channel'])
        self.kernel = SSMKernelDPLR(
            d_model=params['ssm_head'],
            l_max=params['cut_hops'],
            lr=float(params['learning_rate_ssm']),
            init=params['ssm_init']
        )
        self.D = Parameter(torch.randn(params['ssm_head']))
        self.ssm_glu_linear = Linear(params['hidden_channel'], 2 * params['hidden_channel'])
        self.norm1 = LayerNorm(params['hidden_channel'])
        if self.params['ssm_type'] == 'hop_att':
            self.K = Linear(params['hidden_channel'], params['hidden_channel'])
            self.Q = Linear(params['hidden_channel'], params['hidden_channel'])

        # FFN parameters
        self.FFN = Sequential(
            Linear(params['hidden_channel'], params['hidden_channel']),
            GELU(),
            Linear(params['hidden_channel'], params['hidden_channel'])
        )
        self.norm2 = LayerNorm(params['hidden_channel'])

    def forward(self, x, data):
        edge_index_ssm, k_idx = data.edge_index_ssm, data.k_idx
        chunk_size = self.params['hidden_channel']//self.params['ssm_head']

        # compute SSM kernels
        k, k_state = self.kernel(L=k_idx.max().item()+1)
        if self.params['learning_rate_ssm'] == 0:
            k = k.detach()
        k = k.squeeze(0).T

        if ('ablation' in self.params) and (self.params['ablation']!='None'):
            if self.params['ablation'] == 'permuted':
                indices = torch.randperm(k.shape[0]).to(k.device)
                k = k[indices]
            elif self.params['ablation'] == 'random':
                k = torch.randn_like(k).to(k.device)
            elif self.params['ablation'] == 'linear':
                decay = 1./torch.range(1,k.shape[0]).to(k.device)
                k = torch.ones_like(k).to(k.device) * decay.unsqueeze(-1)

        if self.training and self.params['dropout_kernel']!=0:
            mask = torch.rand(k.shape[0], device=k.device) < 1.-self.params['dropout_kernel']
            k = k * (1.0/(1-self.params['dropout_kernel']))
            mask = mask[k_idx]
            edge_index_ssm = edge_index_ssm[:, mask]
            k_idx = k_idx[mask]
        k = k[k_idx]

        # norm, transform
        x_norm = self.norm1(x)
        x_ssm = self.V(x_norm)
        if self.params['ssm_type'] == 'hop_att':
            x_k = self.K(x_norm)
            x_q = self.Q(x_norm)
            # compute multi-head attention
            att_raw = x_k[edge_index_ssm[0]] * x_q[edge_index_ssm[1]]
            att_raw = att_raw.view(-1, self.params['ssm_head'], chunk_size)
            att_raw = att_raw.sum(-1) / math.sqrt(chunk_size)
            # hop-wise normalization
            k_idx_tmp = k_idx + edge_index_ssm[1] * self.params['cut_hops']
            att = softmax(src=att_raw, index=k_idx_tmp)
            assert float('inf') not in att
            k = k * att

        # SSM conv
        x_ssm_mp = x_ssm.view(-1, self.params['ssm_head'], chunk_size)
        x_ssm_mp = self.propagate(edge_index_ssm, x=x_ssm_mp, k=k)
        x_ssm_mp = x_ssm_mp.view(-1, self.params['hidden_channel'])
        x_ssm = x_ssm_mp + x_ssm * self.D.repeat_interleave(chunk_size)
        x_ssm = F.glu(self.ssm_glu_linear(x_ssm))

        # transform, residual
        x = F.dropout(self.O(x_ssm), p=self.params['dropout'], training=self.training) + x

        # FFN
        x = F.dropout(self.FFN(self.norm2(x)), p=self.params['dropout'], training=self.training) + x
        return x

    def message(self, x_j, k):
        return k.unsqueeze(-1) * x_j


# only support hop_att
class S4LightLayer(MessagePassing):
    def __init__(self, params):
        super().__init__(aggr='add', node_dim=0)
        self.params = params
        self.chunk_size = params['hidden_channel']//params['ssm_head']

        # SSM parameters
        self.kernel = SSMKernelDPLR(
            d_model=params['ssm_head'],
            l_max=params['cut_hops'],
            lr=params['learning_rate_ssm'],
            init=params['ssm_init']
        )
        self.D = Parameter(torch.randn(params['ssm_head']))
        self.ssm_glu_linear = Linear(params['hidden_channel'], 2 * params['hidden_channel'])
        self.norm1 = LayerNorm(params['hidden_channel'])

        self.lin_x = Linear(params['hidden_channel'], params['hidden_channel'])
        self.lin_att_source = Parameter(torch.randn(1, params['ssm_head'], self.chunk_size))
        self.lin_att_target = Parameter(torch.randn(1, params['ssm_head'], self.chunk_size))

    def forward(self, x, data):
        edge_index_ssm, k_idx = data.edge_index_ssm, data.k_idx

        # compute SSM kernels
        k, k_state = self.kernel(L=k_idx.max().item()+1)
        k = k.squeeze(0).T
        if self.training and self.params['dropout_kernel']!=0:
            mask = torch.rand(k.shape[0], device=k.device) < 1.-self.params['dropout_kernel']
            k = k * (1.0/(1-self.params['dropout_kernel']))
            mask = mask[k_idx]
            edge_index_ssm = edge_index_ssm[:, mask]
            k_idx = k_idx[mask]
        k = k[k_idx]

        x_ssm = self.lin_x(self.norm1(x))
        x_ssm_mp = x_ssm.view(-1, self.params['ssm_head'], self.chunk_size)
        att_raw_source = (x_ssm_mp * self.lin_att_source).sum(dim=-1)
        att_raw_target = (x_ssm_mp * self.lin_att_target).sum(dim=-1)

        k_idx_tmp = k_idx + edge_index_ssm[1] * self.params['cut_hops']
        k = self.edge_updater(edge_index_ssm, alpha=(att_raw_source, att_raw_target), k=k, k_idx_tmp=k_idx_tmp)

        x_ssm_mp = self.propagate(edge_index_ssm, x=x_ssm_mp, k=k)
        x_ssm_mp = x_ssm_mp.view(-1, self.params['hidden_channel'])
        x_ssm = x_ssm_mp + x_ssm * self.D.repeat_interleave(self.chunk_size)
        x_ssm = F.glu(self.ssm_glu_linear(x_ssm))

        # residual
        x = F.dropout(x_ssm, p=self.params['dropout'], training=self.training) + x

        if self.params['ssm_type'] == 'hop_att':
            return x

    def edge_update(self, alpha_j, alpha_i, k, k_idx_tmp):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        att = softmax(src=alpha, index=k_idx_tmp)
        assert float('inf') not in att
        return k * att

    def message(self, x_j, k):
        return k.unsqueeze(-1) * x_j