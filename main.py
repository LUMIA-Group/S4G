import torch
import wandb
import shutil
import argparse
from utils import get_hash, setup_seed, get_wandb_folder, get_trainer, get_metrics, get_model_size, get_params


if __name__ == '__main__':
    # avoid huge CPU consumption, no affect on speed and performance
    torch.set_num_threads(1)

    # define where wandb logs and models are saved
    parser = argparse.ArgumentParser(description='Parse args from wandb agent command.')
    parser.add_argument('--wandb_folder', '-f', help='Folder for wandb logs and models.', required=True) # wandb, temp
    parser.add_argument('--run_mode', '-r', help='Debug or normal.', required=True) # normal, debug
    args = parser.parse_args()
    dirname = get_wandb_folder(args.wandb_folder)

    if args.run_mode == 'normal':
        wandb.init(dir=dirname, save_code=True, allow_val_change=True, reinit=True, notes='normal')
        wandb.use_artifact('source_code:latest', type='code')
    elif args.run_mode == 'debug':
        wandb.init(dir=dirname, allow_val_change=True, reinit=True, notes='debug', project='S4GNN', config=get_params())

    params_hash = get_hash(dict_in=dict(wandb.config), ignore_keys=['seed', 'repeat_idx', 'split_idx'])
    wandb.config.update({'params_hash': params_hash}) # update params_hash for a run so that we can aggregate different runs with same set of hyper-parameters
    params = dict(wandb.config)

    setup_seed(params['seed'])

    trainer, params = get_trainer(params) # params is updated in get_dataset
    wandb.config.update({'model_size': get_model_size(trainer['model'])})
    print("This trial's parameters: %s"%(params))

    bad_cnt = 0
    best_test_metric = -1e10
    best_val_metric = -1e10
    best_val_loss = 1e10
    time_all = []

    for epoch in range(params['max_epochs']):
        train_metric, train_loss, time_cost_train = get_metrics(trainer, 'train', params)

        if 'tree-neighbors-match' in params['dataset']:
            trainer['scheduler'].step(train_metric)
            val_metric, val_loss, time_cost_val = train_metric, train_loss, time_cost_train
            test_metric, test_loss, time_cost_test = train_metric, train_loss, time_cost_train
            time_all.append(time_cost_train)
        else:
            val_metric, val_loss, time_cost_val = get_metrics(trainer, 'valid', params)
            test_metric, test_loss, time_cost_test = get_metrics(trainer, 'test', params)
            time_all.append(time_cost_train + time_cost_val + time_cost_test)

        if params['stop_item'] == 'metric_val':
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
                bad_cnt = 0
            else:
                bad_cnt += 1
        elif params['stop_item'] == 'loss_val':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_metric = test_metric
                bad_cnt = 0
            else:
                bad_cnt += 1

        if epoch%params['log_freq'] == 0:
            wandb.log({
                'metric/train':train_metric,
                'metric/val':val_metric,
                'metric/test':test_metric,
                'metric/best':best_test_metric,
                'loss/train':train_loss,
                'loss/val':val_loss,
                'loss/test':test_loss,
                'time/train':float(time_cost_train),
                'time/val':float(time_cost_val),
                'time/test':float(time_cost_test)
            })

        print('Metrics: train [%s], val [%s], test [%s], best [%s]'%(train_metric, val_metric, test_metric, best_test_metric))
        
        if bad_cnt == params['patience']:
            break

    print('Final metric is [%s]'%(best_test_metric))
    wandb.run.summary["metric/final"] = best_test_metric
    wandb.run.summary["time/avg"] = sum(time_all)/len(time_all) if len(time_all) != 0 else 0
    wandb.run.summary["time/total"] = sum(time_all) if len(time_all) != 0 else 0

    if args.wandb_folder=='temp':
        shutil.rmtree(dirname)