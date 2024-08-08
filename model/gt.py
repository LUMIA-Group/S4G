from torch.nn import Module
from torch_geometric.nn import GPSConv


class GTBlock(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.gps = GPSConv(
            channels=params['hidden_channel'],
            conv=None,
            heads=4,
            attn_dropout=0.5
        )

    def forward(self, x, edge_attr, data):
        x = self.gps(x=x, edge_index=None, batch=data.batch)
        return x, edge_attr