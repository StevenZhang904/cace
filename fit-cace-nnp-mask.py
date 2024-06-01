#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('cace/')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
PRETRAIN = {"status": True, "ratio": 0.1}
logging.info("Pretraining the model!")
logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='dataset_1593.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 energy_key='energy',
                                 forces_key='force',
                                 atomic_energies={1: -187.6043857100553, 8: -93.80219285502734} # avg
                                 )
cutoff = 5.5
batch_size = 2

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              cutoff=cutoff,
                              pretrain_config=PRETRAIN)

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=4,
                              cutoff=cutoff,
                              pretrain_config=PRETRAIN)

use_device = 'cpu'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


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
                                    forces_key='CACE_disp')

logging.info("building CACE NNP")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)


cace_nnp.to(device)


logging.info(f"First train loop:")


disp_loss = cace.tasks.GetLoss(
    target_name='disp',
    predict_name='CACE_disp',
    loss_fn=nn.CosineSimilarity(),
    loss_weight=1
)

from cace.tools import Metrics

disp_metric = Metrics(
    target_name='disp',
    predict_name='CACE_disp',
    name='disp'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[disp_loss],
        metrics=[disp_metric],
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
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False)

task.save_model('water-model.pth')
cace_nnp.to(device)



trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")



