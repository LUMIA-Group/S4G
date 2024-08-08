import torch


def interval_sampling(data, max_nodes, cutoff_len):
    edge_index_ssm, k_idx, num_nodes = data.edge_index_ssm, data.k_idx, data.num_nodes

    unroll_len = min(num_nodes, cutoff_len)
    if unroll_len > max_nodes:
        # TODO: for now we suppose the graph only contain one connected component
        mask = torch.arange(k_idx.shape[0])
        interval = unroll_len // max_nodes
        mask = mask % unroll_len
        mask = torch.where((mask % interval == 0) & (mask < interval * max_nodes), True, False)

        edge_index_ssm = edge_index_ssm[:, mask]
        k_idx = torch.arange(mask.sum()) % max_nodes
    return edge_index_ssm, k_idx

class IntervalSampling(object):
    def __init__(self, max_nodes, cutoff_len):
        super().__init__()
        self.max_nodes = max_nodes
        self.cutoff_len = cutoff_len

    def __call__(self, data):
        data.edge_index_ssm, data.k_idx = interval_sampling(data, self.max_nodes, self.cutoff_len)
        return data