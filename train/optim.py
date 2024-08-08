# Refer to: https://github.com/HazyResearch/state-spaces/blob/main/example.py


import torch


def setup_optimizer(model, params):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    lr, weight_decay, lr_factor, lr_patience = params['learning_rate'], params['weight_decay'], params['lr_factor'], params['lr_patience']
    if params['stop_item'] == 'metric_val':
        mode = 'max'
    elif params['stop_item'] == 'loss_val':
        mode = 'min'

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    parameters = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    if 'neighbors-match' not in params['dataset']:
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        if params['optimizer'] == 'Adam':
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif params['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        elif params['optimizer'] == 'SGD_momentum':
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif params['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        elif params['optimizer'] == 'Adagrad':
            optimizer = torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
        elif params['optimizer'] == 'Adadelta':
            optimizer = torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        parameters = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": parameters, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=lr_factor, patience=lr_patience, verbose=True, min_lr=0.00001)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))
    return optimizer, scheduler