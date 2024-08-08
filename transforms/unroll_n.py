import torch
from torch_geometric.utils import coalesce, k_hop_subgraph, to_undirected, remove_self_loops, add_self_loops


def unroll(data, params):
    if params['ssm_type'] in ['hop_add', 'hop_att']:
        cutoff_len = params['max_length'] * 2
    elif params['ssm_type'] in ['full_seq']:
        cutoff_len = params['max_length']
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    x = x.float()

    if ('unroll_force_undirected' not in params) or (params['unroll_force_undirected'] == True):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # get all the neighbors for each node (might exceed the cutoff limit), and the order of each neighbor (for sorting later)
    edge_index_ssm = torch.tensor([[], []], dtype=torch.long)
    k_idx = torch.tensor([], dtype=torch.long)

    for node in range(num_nodes):
        subset = torch.tensor([node])
        order_idx = torch.tensor([0])
        order = 0

        # neighbor expansion
        while True:
            subset_new, _, mapping, edge_mask = k_hop_subgraph(
                node_idx=subset,
                num_hops=1,
                edge_index=edge_index,
                num_nodes=num_nodes
            )
            subset_merge = torch.cat([subset, subset_new]).unique()

            if params['ssm_type'] in ['hop_add', 'hop_att']:
                if (subset_new.shape[0] == subset.shape[0]) or (subset_merge.shape[0] > cutoff_len):
                    break
            elif params['ssm_type'] in ['full_seq']:
                if (subset_new.shape[0] == subset.shape[0]) or (subset.shape[0] > cutoff_len):
                    break

            order += 1
            order_idx = torch.cat([
                order_idx,
                torch.ones(subset_new.shape[0] - subset.shape[0], dtype=torch.long) * order
            ])
            mask = torch.isin(subset_new, subset)
            subset = torch.cat([
                subset,
                subset_new[~mask]
            ])

        if params['ssm_type'] in ['full_seq']:
            # computing similarity
            sim = torch.cosine_similarity(x[subset], x[node])
            sim = torch.sigmoid(-sim) # ensure the sim value between 0 and 1, smaller is similar (corresponding to smaller order is closer to the central node)
            score = sim + order_idx
            # sorting
            score, idx = torch.sort(score, stable=True, descending=True)
            subset = subset[idx].flip(dims=[0])
            # generating k_idx and cutting
            subset = subset[:cutoff_len]

        edge_index_ssm = torch.cat(
            [
                edge_index_ssm,
                torch.stack([
                    subset,
                    torch.ones(subset.shape[0], dtype=torch.long) * node
                ])
            ],
            dim=1
        )

        if params['ssm_type'] in ['hop_add', 'hop_att']:
            k_idx = torch.cat([k_idx, order_idx])
        elif params['ssm_type'] in ['full_seq']:
            k_idx = torch.cat([k_idx, torch.arange(subset.shape[0])])

    if params['unroll_loading'] == 'offline':
        return edge_index_ssm, k_idx