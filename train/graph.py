import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset, GNNBenchmarkDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.transforms import AddRandomWalkPE
from sklearn.metrics import average_precision_score, mean_absolute_error, f1_score, accuracy_score
from train.data_utils import get_transformed_data_list, gen_pool_data
from train.eval_utils import get_pred_int, compute_mrr, accuracy_SBM
# from transforms.sampling import IntervalSampling


def get_dataset(trainer, params):
    # transform = IntervalSampling(params['sample_nodes'], params['max_length'])
    hop_edges_cnt = torch.tensor([]).long()
    all_nodes_cnt = 0
    max_nodes_cnt = 0
    for stage in ['train', 'val', 'test']:
        if params['dataset'] in ['peptides-func', 'peptides-struct', 'pascalvoc-sp', 'coco-sp', 'pcqm-contact']:
            dataset = LRGBDataset(root='datasets/pyg/', name=params['dataset'], split=stage)
        elif params['dataset'] in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10']:
            dataset = GNNBenchmarkDataset(root='datasets/pyg/', name=params['dataset'], split=stage)
        elif 'neighbors-match' in params['dataset']:
            depth = int(params['dataset'].split('_')[-1])
            if 'tree' in params['dataset']:
                from train.dataset_dictlookup import DictionaryLookupDataset
                dataset = DictionaryLookupDataset(depth=depth)
                dataset = dataset.generate_data(train_fraction=0.8, depth=depth, stage=stage)
            else:
                from train.dataset_tree import TreeDataset
                path_pt = 'datasets/syntactic/neighbors-match/processed/%s_depth%s.pt'%(stage, depth)
                if not os.path.isfile(path_pt):
                    dataset = getattr(TreeDataset(root='datasets/syntactic/neighbors-match', depth=depth), stage)
                else:
                    dataset = torch.load(path_pt)

        data_list = get_transformed_data_list(params=params, dataset=dataset, stage=stage)
        for data in data_list:
            if params['ssm_type'] in ['hop_add', 'hop_att']:
                all_nodes_cnt += data.num_nodes
                non_neg_idx = torch.where(data.k_idx.long() > -1)
                hop_edges_cnt_tmp = torch.bincount(data.k_idx.long()[non_neg_idx])
                size_tmp = hop_edges_cnt_tmp.shape[0]
                size_cnt = hop_edges_cnt.shape[0]
                if size_tmp > size_cnt:
                    hop_edges_cnt_tmp[:size_cnt] += hop_edges_cnt
                    hop_edges_cnt = hop_edges_cnt_tmp
                else:
                    hop_edges_cnt[:size_tmp] += hop_edges_cnt_tmp
            elif params['ssm_type'] in ['full_seq']:
                max_nodes_cnt = max(max_nodes_cnt, data.num_nodes)

        if ('add_pe' in params) and (params['add_pe'] != 'None'):
            if params['add_pe'] == 'rw':
                transform = AddRandomWalkPE(walk_length=16)
                data_list = [transform(data) for data in data_list]

        trainer[stage+'_loader'] = DataLoader(data_list, batch_size=params['batch_size'], shuffle=(stage=='train'))

    # update params
    if 'neighbors-match' in params['dataset']:
        params['in_channel'] = 2
        params['out_channel'] = 2 ** depth
    elif params['dataset'] in ['pcqm-contact', 'PATTERN']:
        params['in_channel'] = dataset.num_features
        params['out_channel'] = 1
    else:
        params['in_channel'] = dataset.num_features
        params['out_channel'] = dataset.num_classes

    if params['unroll_loading'] == 'online':
        row_pool, col_pool = gen_pool_data(max_num_nodes=512, path_base='datasets/unroll_data')
        params['row_pool'] = row_pool.to(trainer['device'])
        params['col_pool'] = col_pool.to(trainer['device'])

    if params['max_length'] == 'full':
        if params['ssm_type'] in ['hop_add', 'hop_att']:
            params['cut_hops'] = hop_edges_cnt.shape[0]
        elif params['ssm_type'] in ['full_seq']:
            params['cut_hops'] =  max_nodes_cnt
    else:
        if params['ssm_type'] in ['hop_add', 'hop_att']:
            params['cut_hops'] = (hop_edges_cnt.cumsum(0) <= params['max_length'] * all_nodes_cnt).sum().item() + 1
        elif params['ssm_type'] in ['full_seq']:
            params['cut_hops'] = params['max_length']
    return trainer, params


def run(trainer, stage, params):
    data_, train_loader, val_loader, test_loader, device, model, optimizer, scheduler, evaluator_ = trainer.values()

    # set model mode
    if stage == 'train':
        torch.set_grad_enabled(True)
        model.train()
        loader = train_loader
    else:
        torch.set_grad_enabled(False)
        model.eval()
        if stage == 'valid':
            loader = val_loader
        elif stage == 'test':
            loader = test_loader

    total_loss = 0
    N = 0
    y_preds, y_trues = [], []

    for data in loader:
        # setup data
        if (stage == 'train') and (params['pe'] == 'lap'):
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)

        # get node-level representation
        out = model(data)

        # pool node features to get task-specific level representation
        if params['pool_head'] == 'graph':
            out = global_mean_pool(out, data.batch)
        elif params['pool_head'] == 'link':
            data.x = out
            out = (out[data.edge_label_index[0]] * out[data.edge_label_index[1]]).sum(dim=-1)
        elif params['pool_head'] == 'root':
            out = out[data.root_mask]

        # compute loss
        if params['dataset'] in ['peptides-func']:
            loss = F.binary_cross_entropy_with_logits(out, data.y)
        elif params['dataset'] in ['peptides-struct']:
            loss = F.l1_loss(out, data.y)
        elif params['dataset'] in ['pascalvoc-sp', 'coco-sp', 'PATTERN', 'CLUSTER']:
            out = out.squeeze(-1)
            # calculating label weights for weighted loss computation
            V = data.y.size(0)
            n_classes = out.shape[1] if out.ndim > 1 else 2
            label_count = torch.bincount(data.y)
            label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
            cluster_sizes = torch.zeros(n_classes, device=data.y.device).long()
            cluster_sizes[torch.unique(data.y)] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes > 0).float()
            if out.ndim > 1: # multiclass
                out = F.log_softmax(out, dim=-1)
                loss = F.nll_loss(out, data.y, weight=weight)
            else: # binary
                loss = F.binary_cross_entropy_with_logits(out, data.y.float(), weight=weight[data.y])
                out = torch.sigmoid(out)
            out = get_pred_int(out)
        elif params['dataset'] in ['MNIST', 'CIFAR10']:
            out = F.log_softmax(out, dim=-1)
            loss = F.nll_loss(out, data.y)
            out = get_pred_int(out)
        elif params['dataset'] in ['pcqm-contact']:
            loss = F.binary_cross_entropy_with_logits(out, data.edge_label.float())
            data.edge_index_ssm = data.k_idx = None
            data.y = torch.tensor([compute_mrr(data)['mrr']]) # MRR for this batch
            out = torch.tensor([data.edge_label.shape[0]]) # num of labels in this batch
        elif 'neighbors-match' in params['dataset']:
            data.y -= 1
            loss = F.cross_entropy(out, data.y)
            _, out = out.max(dim=1)
        y_trues.append(data.y.detach())
        y_preds.append(out.detach())

        # loss backward (optional)
        if stage == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs

    loss = total_loss / N

    # compute metric
    y_preds = torch.cat(y_preds).cpu().numpy()
    y_trues = torch.cat(y_trues).cpu().numpy()
    if params['dataset'] in ['peptides-func']: # AP
        metric = average_precision_score(y_trues, y_preds)
    elif params['dataset'] in ['peptides-struct']: # MAE
        metric = -mean_absolute_error(y_trues, y_preds)
    elif params['dataset'] in ['pascalvoc-sp', 'coco-sp']: # F1
        metric = f1_score(y_trues, y_preds, average='macro', zero_division=0)
    elif params['dataset'] in ['pcqm-contact']: # MRR
        metric = (y_trues * y_preds).sum() / y_preds.sum()
    elif ('neighbors-match' in params['dataset']) or (params['dataset'] in ['MNIST', 'CIFAR10']): # ACC
        metric = accuracy_score(y_trues, y_preds)
    elif params['dataset'] in ['PATTERN', 'CLUSTER']: # ACC-SBM
        metric = accuracy_SBM(np.squeeze(y_trues), y_preds)

    # scheduler update (optional)
    if (stage == 'valid') and (scheduler != None):
        if params['stop_item'] == 'metric_val':
            scheduler.step(metric)
        elif params['stop_item'] == 'loss_val':
            scheduler.step(loss)
    return metric, loss