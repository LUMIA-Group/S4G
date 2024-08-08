from torch.nn import Module, Linear, ModuleList, Embedding
from train.data_utils import transform_precision, gen_unroll_data, cut_data
from model.s4g import S4GBlock
from model.gt import GTBlock


class GraphModel(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        if params['input_encoder'] == 'AtomEncoder':
            from ogb.graphproppred.mol_encoder import AtomEncoder
            self.input_encoder = AtomEncoder(params['hidden_channel'])
        elif params['input_encoder'] == 'LinearEncoder':
            self.input_encoder = Linear(params['in_channel'], params['hidden_channel'])
        elif params['input_encoder'] == 'DiscreteEncoder': # only works for neighbors-match dataset for now
            self.input_encoder_k = Embedding(params['out_channel']+1, params['hidden_channel'])
            self.input_encoder_v = Embedding(params['out_channel']+1, params['hidden_channel'])

        if params['edge_encoder'] == 'BondEncoder':
            from ogb.graphproppred.mol_encoder import BondEncoder
            self.edge_encoder = BondEncoder(params['hidden_channel'])
        elif params['edge_encoder'] == 'LinearEncoder':
            if params['dataset'] in ['MNIST', 'CIFAR10']:
                self.edge_encoder = Linear(1, params['hidden_channel'])
            elif params['dataset'] in ['pascalvoc-sp', 'coco-sp']:
                self.edge_encoder = Linear(2, params['hidden_channel'])

        if params['pe'] in ['rw', 'lap']:
            self.pe_encoder = Linear(params[params['pe']+'_dim'], params['hidden_channel'])
        if ('add_pe' in params) and (params['add_pe'] != 'None'):
            if params['add_pe'] == 'rw':
                self.pe_encoder = Linear(16, params['hidden_channel'])

        if ('tree-neighbors-match' in params['dataset']) and (params['num_layers']) == 'None':
            num_layers = int(params['dataset'].split('_')[-1])
        else:
            num_layers = params['num_layers']

        self.encoders = ModuleList()
        for _ in range(num_layers):
            if self.params['model'] == 'S4G':
                self.encoders.append(S4GBlock(params))
            elif self.params['model'] == 'GT':
                self.encoders.append(GTBlock(params))

        self.predictor = Linear(params['hidden_channel'], params['out_channel'])

    def forward(self, data):
        if self.params['task_type'] == 'graph':
            if self.params['unroll_loading'] == 'online':
                data = gen_unroll_data(data=data, row_pool=self.params['row_pool'], col_pool=self.params['col_pool'])
            elif self.params['unroll_loading'] == 'offline':
                data = transform_precision(data=data)

        if self.params['max_length'] != 'full':
            data = cut_data(data=data, cut_hops=self.params['cut_hops']) # only keep the nodes in the rooted-tree with specific height

        x = data.x
        if 'neighbors-match' in self.params['dataset']:
            x = self.input_encoder_k(x[:, 0]) + self.input_encoder_v(x[:, 1])
        else:
            x = self.input_encoder(x)

        # get pe
        if self.params['pe'] in ['rw', 'lap']:
            if self.params['pe'] == 'rw':
                pe = data.rw_pos_enc
            elif self.params['pe'] == 'lap':
                pe = data.lap_pos_enc
            x += self.pe_encoder(pe)
        if ('add_pe' in self.params) and (self.params['add_pe'] != 'None'):
            if self.params['add_pe'] == 'rw':
                x += self.pe_encoder(data.random_walk_pe)

        # update edge_attr
        if self.params['edge_encoder'] != 'None':
            if self.params['dataset'] in ['MNIST', 'CIFAR10']:
                edge_attr = self.edge_encoder(data.edge_attr.view(-1, 1))
            else:
                edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None

        for encoder in self.encoders:
            x, edge_attr = encoder(x, edge_attr, data)

        x = self.predictor(x)
        return x