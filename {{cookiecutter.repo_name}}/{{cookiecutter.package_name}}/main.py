import os
import random
from typing import Any, List, Tuple, Dict
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler

import {{ cookiecutter.package_name }}.data_loader.augmentation as module_aug
import {{ cookiecutter.package_name }}.data_loader.data_loaders as module_data
import {{ cookiecutter.package_name }}.model.loss as module_loss
import {{ cookiecutter.package_name }}.model.metric as module_metric
import {{ cookiecutter.package_name }}.model.model as module_arch
from {{ cookiecutter.package_name }}.trainer import Trainer
from {{ cookiecutter.package_name }}.utils import setup_logger


log = setup_logger(__name__)


def train(cfg: Dict, resume: str) -> None:
    log.debug(f'Training: {cfg}')
    seed_everything(cfg['seed'])

    model = get_instance(module_arch, 'arch', cfg)
    model, device = setup_device(model, cfg['target_devices'])
    torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

    param_groups = setup_param_groups(model, cfg['optimizer'])
    optimizer = get_instance(module_optimizer, 'optimizer', cfg, param_groups)
    lr_scheduler = get_instance(module_scheduler, 'lr_scheduler', cfg, optimizer)
    model, optimizer, start_epoch = resume_checkpoint(resume, model, optimizer, cfg)

    transforms = get_instance(module_aug, 'augmentation', cfg)
    data_loader = get_instance(module_data, 'data_loader', cfg, transforms)
    valid_data_loader = data_loader.split_validation()

    log.info('Getting loss and metric function handles')
    loss = getattr(module_loss, cfg['loss'])
    metrics = [getattr(module_metric, met) for met in cfg['metrics']]

    log.info('Initialising trainer')
    trainer = Trainer(model, loss, metrics, optimizer,
                        start_epoch=start_epoch,
                        config=cfg,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    trainer.train()
    log.info('Finished!')


def setup_device(
    model: nn.Module,
    target_devices: List[int]
) -> Tuple[torch.device, List[int]]:
    """
    setup GPU device if available, move model into configured device
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        log.warning(
            "There's no GPU available on this machine. Training will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    if not target_devices:
        log.info("No GPU selected. Training will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu} "
                "available. Check the configuration and try again.")
        log.critical(msg)
        raise Exception(msg)

    log.info(f'Using devices {target_devices} of available devices {available_devices}')
    device = torch.device(f'cuda:{target_devices[0]}')
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices)
    else:
        model = model.to(device)
    return model, device


def setup_param_groups(model: nn.Module, config: Dict) -> List:
    return [{'params': model.parameters(), **config}]


def resume_checkpoint(resume_path, model, optimizer, config):
    """
    Resume from saved checkpoint.
    """
    if not resume_path:
        return model, optimizer, 0

    log.info(f'Loading checkpoint: {resume_path}')
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != config['optimizer']['type']:
        log.warning("Warning: Optimizer type given in config file is different from "
                            "that of checkpoint. Optimizer parameters not being resumed.")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])

    log.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, checkpoint['epoch']


def get_instance(
    module: ModuleType,
    name: str,
    config: Dict,
    *args: Any
) -> Any:
    """
    Helper to construct an instance of a class.

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    name : str
        Name of class, as would be returned by ``.__class__.__name__``.
    config : dict
        Dictionary containing an 'args' item, which will be used as ``kwargs`` to construct the
        class instance.
    args : Any
        Positional arguments to be given before ``kwargs`` in ``config``.
    """
    ctor_name = config[name]['type']
    log.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(*args, **config[name]['args'])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
