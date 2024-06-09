#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('cace/')
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import wandb
import yaml
import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)
cace.tools.setup_logger(level='INFO')

TRAIN_FROM_SCRATCH = True
PRETRAIN_CKPT_PATH = "pretrain-water-model-0.5-epoch40.pth"
use_device = 'cuda:0'

if TRAIN_FROM_SCRATCH:
    Hyperparams = {}
    Hyperparams['train_from_scratch'] = TRAIN_FROM_SCRATCH
else:
    config_path = 'pretrain_hyperparams.yaml'
    if os.path.exists(config_path):
        with open (config_path, 'r') as file:
            Hyperparams = yaml.load(file, Loader=yaml.FullLoader)
            Hyperparams['train_from_scratch'] = TRAIN_FROM_SCRATCH
            Hyperparams['PRETRAIN_CKPT_PATH'] = PRETRAIN_CKPT_PATH
    else:
        Hyperparams = {}
        Hyperparams['train_from_scratch'] = TRAIN_FROM_SCRATCH
        Hyperparams['PRETRAIN_CKPT_PATH'] = PRETRAIN_CKPT_PATH
        
wandb.init(project='CACE', config=Hyperparams, notes='No restart training')

PRETRAIN = {"status": False, "ratio": 0}


logging.info("Finetuining the model!")
logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='dataset_1593.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 energy_key='energy',
                                 forces_key='force',
                                 atomic_energies={1: -187.6043857100553, 8: -93.80219285502734} # avg
                                 )
cutoff = 5.5
batch_size = 20

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              cutoff=cutoff,
                              pretrain_config=PRETRAIN,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=4,
                              cutoff=cutoff,
                              pretrain_config=PRETRAIN, 
                              )

device = cace.tools.init_device(use_device)
# device = torch.device(use_device)
logging.info(f"device: {device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1,8],
    n_atom_basis=3,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    avg_num_neighbors=1,
    device=device,
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)


forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

logging.info("building CACE NNP")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)
# load state_dict from pre-trained model
if Hyperparams['train_from_scratch'] == False:
    if os.path.exists(PRETRAIN_CKPT_PATH):
        cace_nnp = torch.load(PRETRAIN_CKPT_PATH)
        logging.info("Pre-trained model loaded")
    else:
        logging.info("Pre-trained model not found")

cace_nnp.to(device)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     cace_nnp = nn.DataParallel(cace_nnp)

logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

task = TrainingTask(
    model=cace_nnp,
    losses=[energy_loss, force_loss],
    metrics=[e_metric, f_metric],
    device=device,
    optimizer_args=optimizer_args,
    scheduler_cls=torch.optim.lr_scheduler.StepLR,
    scheduler_args=scheduler_args,
    max_grad_norm=10,
    ema=True,
    ema_start=10,
    warmup_steps=5,
)

logging.info("training")
task.fit(train_loader, valid_loader, epochs=300, screen_nan=False)

task.save_model('water-model.pth')


logging.info(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")



