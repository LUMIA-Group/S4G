import torch.nn.functional as F
from torch.nn import Module, Linear, Sequential, Dropout, GELU, LayerNorm
from torch_geometric.nn import global_mean_pool
from layer.s4_layer import S4Layer, S4LightLayer


class S4GBlock(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        if params['seq_encoder'] == 'S4':
            self.ssm_conv = S4Layer(params)
        elif params['seq_encoder'] == 'S4Light':
            self.ssm_conv = S4LightLayer(params)

        if params['graph_encoder'] == 'GCN':
            from torch_geometric.nn import GCNConv
            self.mp_conv = GCNConv(
                in_channels=params['hidden_channel'],
                out_channels=params['hidden_channel']
            )
            self.norm3 = LayerNorm(params['hidden_channel'])
        elif params['graph_encoder'] == 'GatedGCN':
            from layer.gatedgcn_layer import GatedGCNLayer
            self.mp_conv = GatedGCNLayer(
                in_dim=params['hidden_channel'],
                out_dim=params['hidden_channel'],
                dropout=params['dropout'],
                residual=True
            )
            self.norm3 = LayerNorm(params['hidden_channel'])
            self.norm4 = LayerNorm(params['hidden_channel'])

        if (params['seq_encoder'] != 'None') and (params['graph_encoder'] != 'None'):
            self.fusing_mlp = Sequential(
                Linear(params['hidden_channel'], params['hidden_channel'] * 2),
                GELU(),
                Dropout(params['dropout']),
                Linear(params['hidden_channel'] * 2, params['hidden_channel']),
                Dropout(params['dropout']),
            )
            self.fusing_norm = LayerNorm(params['hidden_channel'])

    def forward(self, x, edge_attr, data):
        # S4
        if self.params['seq_encoder'] in ['S4', 'S4Light']:
            if ('involve_edge' in self.params) and (self.params['involve_edge'] == True):
                e = global_mean_pool(x=edge_attr, batch=data.edge_index[1], size=data.num_nodes)
                x_ssm = self.ssm_conv(x+e, data)
            else:
                x_ssm = self.ssm_conv(x, data)

        # MPNN
        if self.params['graph_encoder'] == 'GCN':
            x_mp = self.norm3(x)
            x_mp = self.mp_conv(x=x_mp, edge_index=data.edge_index)
            x_mp = F.relu(x_mp)
            x_mp = F.dropout(x_mp, p=self.params['dropout'], training=self.training) + x
        elif self.params['graph_encoder'] == 'GatedGCN':
            x_mp = self.norm3(x)
            edge_attr = self.norm4(edge_attr)
            x_mp, edge_attr = self.mp_conv(x=x_mp, e=edge_attr, edge_index=data.edge_index)

        if self.params['graph_encoder'] == 'None':
            x = x_ssm
        elif self.params['seq_encoder'] == 'None':
            x = x_mp
        else:
            x = x_ssm + x_mp
            x = self.fusing_mlp(self.fusing_norm(x)) + x

        return x, edge_attr