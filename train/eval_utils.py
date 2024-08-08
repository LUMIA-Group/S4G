import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def get_pred_int(pred_score):
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        return (pred_score > 0.5).long()
    else:
        return pred_score.max(dim=1)[1]


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


# Refer to: https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/head/inductive_edge.py
def compute_mrr(batch):
    stats = {}
    for data in batch.to_data_list():
        # print(data.num_nodes)
        # print(data.edge_index_labeled)
        # print(data.edge_label)
        pred = data.x @ data.x.transpose(0, 1)
        # print(pred.shape)

        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]
        # print(pos_edge_index, num_pos_edges)

        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]
        # print(pred_pos)

        if num_pos_edges > 0:
            neg_mask = torch.ones([num_pos_edges, data.num_nodes],
                                    dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            # print(pred_neg, pred_neg.shape)
            mrr_list = _eval_mrr(pred_pos, pred_neg, 'torch')
        else:
            # Return empty stats.
            mrr_list = _eval_mrr(pred_pos, pred_pos, 'torch')

        # print(mrr_list)
        for key, val in mrr_list.items():
            if key.endswith('_list'):
                key = key[:-len('_list')]
                val = float(val.mean().item())
            if np.isnan(val):
                val = 0.
            if key not in stats:
                stats[key] = [val]
            else:
                stats[key].append(val)
            # print(key, val)
        # print('-' * 80)

    # print('=' * 80, batch.split)
    batch_stats = {}
    for key, val in stats.items():
        mean_val = sum(val) / len(val)
        batch_stats[key] = mean_val
        # print(f"{key}: {mean_val}")
    return batch_stats


# Refer to: https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/head/inductive_edge.py
def _eval_mrr(y_pred_pos, y_pred_neg, type_info):
    """ Compute Hits@k and Mean Reciprocal Rank (MRR).

    Implementation from OGB:
    https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

    Args:
        y_pred_neg: array with shape (batch size, num_entities_neg).
        y_pred_pos: array with shape (batch size, )
    """

    if type_info == 'torch':
        y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
        argsort = torch.argsort(y_pred, dim=1, descending=True)
        ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
        ranking_list = ranking_list[:, 1] + 1
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1. / ranking_list.to(torch.float)

        return {'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@10_list': hits10_list,
                'mrr_list': mrr_list}

    else:
        y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg],
                                axis=1)
        argsort = np.argsort(-y_pred, axis=1)
        ranking_list = (argsort == 0).nonzero()
        ranking_list = ranking_list[1] + 1
        hits1_list = (ranking_list <= 1).astype(np.float32)
        hits3_list = (ranking_list <= 3).astype(np.float32)
        hits10_list = (ranking_list <= 10).astype(np.float32)
        mrr_list = 1. / ranking_list.astype(np.float32)

        return {'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@10_list': hits10_list,
                'mrr_list': mrr_list}