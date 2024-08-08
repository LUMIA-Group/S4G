import torch
import hashlib
import random
import json
import time
import tempfile
import os
import importlib
import yaml
import numpy as np
from train.optim import setup_optimizer


def get_trainer(params):
    vars_list = ['data', 'train_loader', 'val_loader', 'test_loader', 'device', 'model', 'optimizer', 'scheduler', 'evaluator']
    trainer = {item:None for item in vars_list}

    # get device
    trainer['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get dataset
    get_dataset = importlib.import_module('train.%s'%(params['task_type'])).get_dataset
    trainer, params = get_dataset(trainer, params) # params is updated

    # get model
    from model.framework import GraphModel
    trainer['model'] = GraphModel(params).to(trainer['device'])

    # get optimizer and scheduler
    trainer['optimizer'], trainer['scheduler'] = setup_optimizer(trainer['model'], params)
    return trainer, params


def get_metrics(trainer, stage, params):
    start_time = time.time()

    run = importlib.import_module('train.%s'%(params['task_type'])).run
    metric, loss = run(trainer, stage, params)

    end_time = time.time()
    time_cost = end_time-start_time
    return metric, loss, time_cost


# generate hash tag for one set of hyper parameters
def get_hash(dict_in, ignore_keys):
    dict_in = {k:v for k,v in dict_in.items() if k not in ignore_keys}
    hash_out = hashlib.blake2b(json.dumps(dict_in, sort_keys=True).encode(), digest_size=4).hexdigest()
    return str(hash_out)


def get_wandb_folder(path_type):
    if path_type=='temp':
        folder_temp = tempfile.TemporaryDirectory()
        tmpdirname = folder_temp.name
        os.chmod(tmpdirname, 0o777)
        dir_name = tmpdirname
    elif path_type=='wandb':
        dir_name = 'wandb'
    return dir_name


def get_timestamp():
    time.tzset()
    now = int(round(time.time()*1000))
    timestamp = time.strftime('%Y-%m%d-%H%M',time.localtime(now/1000))
    return timestamp


def get_params(path_config_yaml='configs.yaml'):
    file = open(path_config_yaml, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    params = yaml.load(file_data, Loader=yaml.Loader)['parameters']
    for k in params:
        params[k] = params[k]['values'][0]
    return params


def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# fix random seed
def setup_seed(seed):
    if seed != 'None':
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True