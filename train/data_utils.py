import os
import torch
from torch_geometric.utils import mask_to_index, is_undirected


def generate_splits(data, g_split):
  n_nodes = len(data.x)
  train_mask = torch.zeros(n_nodes, dtype=bool)
  valid_mask = torch.zeros(n_nodes, dtype=bool)
  test_mask = torch.zeros(n_nodes, dtype=bool)
  idx = torch.randperm(n_nodes)
  val_num = test_num = int(n_nodes * (1 - g_split) / 2)
  train_mask[idx[val_num + test_num:]] = True
  valid_mask[idx[:val_num]] = True
  test_mask[idx[val_num:val_num + test_num]] = True
  data.train_mask = train_mask
  data.val_mask = valid_mask
  data.test_mask = test_mask
  return data


def cut_data(data, cut_hops):
    edge_index_ssm = data.edge_index_ssm
    k_idx = data.k_idx
    mask = torch.where(k_idx < cut_hops)
    data.edge_index_ssm = edge_index_ssm[:, mask[0]]
    data.k_idx = k_idx[mask]
    return data


def gen_pool_data(max_num_nodes, path_base):
    os.makedirs(path_base, exist_ok=True)
    path_row = os.path.join(path_base, 'row_pool_%s.pt'%(max_num_nodes))
    path_col = os.path.join(path_base, 'col_pool_%s.pt'%(max_num_nodes))

    if not (os.path.isfile(path_row) and os.path.isfile(path_col)):
        row_pool = []
        col_pool = []
        for i in range(max_num_nodes):
            tmp = torch.arange(max_num_nodes)
            tmp[i:] = -1
            row = tmp.repeat(max_num_nodes)
            row[i * max_num_nodes:] = -1
            row_pool.append(row)
            col = tmp.repeat_interleave(max_num_nodes)
            mask = torch.where(row == -1)
            col[mask] = -1
            col_pool.append(col)
        row = torch.stack(row_pool).type(torch.int16)
        col = torch.stack(col_pool).type(torch.int16)
        torch.save(row, path_row)
        torch.save(col, path_col)
    else:
        row = torch.load(path_row)
        col = torch.load(path_col)
    return row, col


def gen_unroll_data(data, row_pool, col_pool):
    graph_num_nodes = data.ptr[1:] - data.ptr[:-1]
    offset = data.ptr[:-1].unsqueeze(-1)
    row_tmp = row_pool[graph_num_nodes]
    col_tmp = col_pool[graph_num_nodes]
    mask = torch.where(row_tmp > -1)
    row_tmp = (row_tmp + offset)[mask].flatten().long()
    col_tmp = (col_tmp + offset)[mask].flatten().long()
    edge_index_ssm = torch.stack([row_tmp, col_tmp])

    k_idx = data.k_idx.long()
    mask = torch.where(k_idx > -1)
    k_idx = k_idx[mask]
    edge_index_ssm = edge_index_ssm[:, mask[0]]

    data.edge_index_ssm = edge_index_ssm
    data.k_idx = k_idx
    return data


def transform_precision(data):
    data.k_idx = data.k_idx.long()
    data.adj_ssm = data.adj_ssm.T.long()

    mask = (data.adj_ssm[1]==0) & (data.k_idx==0)
    segment = mask_to_index(mask) # [batch_size]

    ptr = data.ptr.new_zeros(data.ptr.shape[0]-1)
    ptr[1:] = (data.ptr[1:] - data.ptr[:-1])[:-1]

    offset = data.k_idx.new_zeros(data.k_idx.shape)
    offset[segment] = ptr

    data.edge_index_ssm = data.adj_ssm + offset.cumsum(0)
    data.adj_ssm = None
    return data


# node style task
def get_transformed_data(params, data):
    from transforms.unroll_n import unroll
    path_base = 'datasets/unroll_data/%s'%(params['dataset'])
    os.makedirs(path_base, exist_ok=True)
    if params['ssm_type'] in ['hop_add', 'hop_att']:
        path_edge_index_ssm = os.path.join(path_base, 'edge_index_ssm_%sx2.pt'%(params['max_length']))
        path_k_idx = os.path.join(path_base, 'k_idx_%sx2.pt'%(params['max_length']))
    elif params['ssm_type'] in ['full_seq']:
        path_edge_index_ssm = os.path.join(path_base, 'edge_index_ssm_%s.pt'%(params['max_length']))
        path_k_idx = os.path.join(path_base, 'k_idx_%s.pt'%(params['max_length']))

    if not is_undirected(data.edge_index):
        if ('unroll_force_undirected' in params) and (params['unroll_force_undirected'] == False):
            path_edge_index_ssm = path_edge_index_ssm.replace('.pt', '-directed.pt')
            path_k_idx = path_k_idx.replace('.pt', '-directed.pt')

    if not (os.path.isfile(path_edge_index_ssm) and os.path.isfile(path_k_idx)): # transform and save
        print('Processing...')
        edge_index_ssm, k_idx = unroll(data=data, params=params)
        torch.save(edge_index_ssm, path_edge_index_ssm)
        torch.save(k_idx, path_k_idx)
        print('Done!')

    # load and ensemble
    data.edge_index_ssm = torch.load(path_edge_index_ssm)
    data.k_idx = torch.load(path_k_idx)
    return data


# graph style task
def get_transformed_data_list(params, dataset, stage):
    from transforms.unroll_g import unroll
    tensor_type = {'int16': torch.int16, 'int8': torch.int8}
    path_base = 'datasets/unroll_data/%s'%(params['dataset'])
    os.makedirs(path_base, exist_ok=True)
    path_edge_index_ssm = os.path.join(path_base, 'edge_index_ssm_%s_int16_non-neg.pt'%(stage))
    if params['ssm_type'] in ['hop_add', 'hop_att']:
        if params['unroll_loading'] == 'offline':
            path_k_idx = os.path.join(path_base, 'k_idx_%s_int8_non-neg.pt'%(stage))
        elif params['unroll_loading'] == 'online':
            path_k_idx = os.path.join(path_base, 'k_idx_%s_int8_has-neg.pt'%(stage))
    elif params['ssm_type'] in ['full_seq']:
        if (params['unroll_loading'] == 'offline') or ('neighbors-match' in params['dataset']):
            path_k_idx = os.path.join(path_base, 'k_idx_%s_int16_non-neg.pt'%(stage))
    if params['dataset'] in ['peptides-func', 'peptides-struct']:
        path_k_idx = path_k_idx.replace('int8', 'int16')
    data_list = []

    if not is_undirected(dataset[0].edge_index):
        if ('unroll_force_undirected' in params) and (params['unroll_force_undirected'] == False):
            path_edge_index_ssm = path_edge_index_ssm.replace('.pt', '-directed.pt')
            path_k_idx = path_k_idx.replace('.pt', '-directed.pt')

    # transform and save
    if 'neighbors-match' in params['dataset']:
        if params['unroll_loading'] == 'online':
            print('Processing...')
            edge_index_ssm, k_idx = unroll(data=dataset[0], params=params)
            edge_index_ssm = edge_index_ssm.type(torch.int16).T # avoid offset when load by DataLoader
            k_idx = k_idx.type(tensor_type[path_k_idx.split('_')[-2]])
            for _, data in enumerate(dataset):
                data.k_idx = k_idx
                data_list.append(data)
            print('Done!')
    else:
        if not os.path.isfile(path_k_idx):
            print('Processing...')
            edge_index_ssm_list, k_idx_list = [], []
            for idx, data in enumerate(dataset):
                print('Processing No.%s of %s data...'%(idx+1, len(dataset)))
                edge_index_ssm, k_idx = unroll(data=data, params=params)
                if not os.path.isfile(path_edge_index_ssm) and params['unroll_loading'] == 'offline':
                    edge_index_ssm_list.append(edge_index_ssm.type(torch.int16))
                k_idx_list.append(k_idx.type(tensor_type[path_k_idx.split('_')[-2]]))
            if not os.path.isfile(path_edge_index_ssm) and params['unroll_loading'] == 'offline':
                torch.save(edge_index_ssm_list, path_edge_index_ssm)
            torch.save(k_idx_list, path_k_idx)
            print('Done!')
        # load and ensemble
        if params['unroll_loading'] == 'offline':
            edge_index_ssm_list = torch.load(path_edge_index_ssm)
        k_idx_list = torch.load(path_k_idx)
        for idx, data in enumerate(dataset):
            if params['unroll_loading'] == 'offline':
                data.adj_ssm = edge_index_ssm_list[idx].T # avoid offset when load by DataLoader
            data.k_idx = k_idx_list[idx]
            data_list.append(data)
    return data_list