import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, to_undirected, remove_self_loops


def expand(edge_index, num_nodes):
    adj_t = SparseTensor.from_edge_index(edge_index=edge_index, sparse_sizes=(num_nodes, num_nodes))

    ranking_bias = torch.eye(num_nodes) - 1 # for final ranking, assign low score to low-order neighbors
    bias = 1
    mask = torch.eye(num_nodes).bool() # initial seed nodes

    flag = True
    while flag:
        res = adj_t @ mask.T.float()
        res = res.T.bool()

        mask_new = mask | res
        hop_nodes = torch.bitwise_xor(mask_new, mask)

        if hop_nodes.sum() == 0:
            flag = False
        else:
            ranking_bias[hop_nodes] = bias
            mask = mask_new
            bias += 1

    positive_mask = torch.where(ranking_bias>0)
    return ranking_bias, positive_mask


def unroll(data, params):
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    x = x.float()

    if ('unroll_force_undirected' not in params) or (params['unroll_force_undirected'] == True):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index)

    # neighbor expansion
    ranking_bias, positive_mask = expand(edge_index, num_nodes)

    if params['ssm_type'] in ['hop_add', 'hop_att']:
        idx = torch.where(ranking_bias >= -1)
        edge_index_ssm = torch.stack([idx[1], idx[0]], dim=0)
        k_idx = ranking_bias.T.flatten().long()
        if params['unroll_loading'] == 'offline':
            non_neg_idx = torch.where(k_idx > -1)
            edge_index_ssm = edge_index_ssm[:, non_neg_idx[0]]
            k_idx = k_idx[non_neg_idx[0]]
    elif params['ssm_type'] in ['full_seq']:
        if (params['unroll_loading'] == 'offline') or ('neighbors-match' in params['dataset']):
            # computing similarity
            score = torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
            score = torch.sigmoid(-score)
            ranking_bias[positive_mask] = score[positive_mask] + ranking_bias[positive_mask]

            # sorting
            sorted, seqs = torch.sort(ranking_bias, stable=True, descending=True)
            negtive_mask = torch.where(sorted<0)
            seqs[negtive_mask] = -1

            # generating k_idx and cutting
            seqs_lens = num_nodes - (seqs==-1).sum(-1) # get the length of each sequence
            edge_index_ssm = torch.stack([
                torch.cat([seqs[i][:seqs_lens[i]].flip(dims=[0]) for i in range(num_nodes)]),
                torch.cat([torch.ones(seqs_lens[i], dtype=torch.long) * i for i in range(num_nodes)])
            ])
            k_idx = torch.cat([torch.arange(seqs_lens[i]) for i in range(num_nodes)])
    return edge_index_ssm, k_idx