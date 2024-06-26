import os
import sys
import tempfile
import math

import numpy as np

import wandb

import torch
import torch.nn as nn
import pytorch_lightning as pl

import torchmetrics
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import R2Score

# Custom imports
sys.path.append('/mnt/src/')
import learning.data_module
import utils

"""
We want to learn how the dynamics of a system evolve over time
in latent space

The first part is an autoencoder which learns a compressed
representation of the input data

The second part takes compressed representations and predicts
the next state in the sequence (in latent space), and then
decodes this to the original space using the decoder from
the autoencoder, and compares this to the ground truth
to create a learning signal

Generally the input data will be a history of state, action,
state, action, ... and the output will be the next state

Autoencoder: S_i -> (encoder) -> Z_i -> (decoder) -> S_i
Dynamics: Z_i and a_i -> (dynamics) -> Z_{i+1} -> (decoder) -> S_{i+1}

So the encoder and decoder is trained on reconstruction loss
and the dynamics model is trained on prediction loss, using the frozen
decoder    

We can then quickly simulate the dynamics of the system by 
running the current state with some action to get the subsequent 
state, and then repeat this process

For this reason we don't want the autoencoder to be dependent
on the actions, only the states
"""

class Autoencoder(pl.LightningModule):
    def __init__(
        self, 
        lr=0.001,
    ):
        super().__init__()
         
        # Save the inputs
        self.lr = lr
        
        # Convienent for determining device
        self._dummy_param = nn.Parameter(torch.empty(0))

        # Model is a simple autoencoder
        self.dim_state  = utils.DIM_STATE
        self.dim_latent = utils.DIM_LATENT
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_state, 16),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            nn.Linear(16, self.dim_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_latent, 16),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            nn.Linear(16, self.dim_state),
        )

        # Loss function is MSE
        self.loss_computer = nn.MSELoss()
        
    def get_device(self):
        return self._dummy_param.device
    
    def forward(self, batch):
        # Even though it's just one thing we still need to unpack
        x = batch[0]
        # Encode and decode
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def _train_val_test_helper(self, batch, batch_idx, name):
        """
        This function does a forward pass, computes the loss,
        and logs the loss (and other quantities of interest)
        """
        
        def simple_log(quantity_name, quantity):
            self.log(
                f"{name}_{quantity_name}", 
                quantity, 
                prog_bar=True,
                logger=True, 
                #on_step=True, 
                on_epoch=True,
            )

        # Extract inputs and prediction targets from batch
        x = batch[0]
        x_hat, z = self.forward(batch)
        x_hat = x_hat.view_as(x)
        
        # Compute, log, and return the loss
        loss = self.loss_computer(x, x_hat) 
        simple_log("loss (mse)", loss)
        return {
            "loss": loss
        }

    def training_step(self, batch, batch_idx):
        return self._train_val_test_helper(
            batch, batch_idx, "train"
        )

    def validation_step(self, batch, batch_idx):
        return self._train_val_test_helper(
            batch, batch_idx, "val"
        )

    def test_step(self, batch, batch_idx):        
        return self._train_val_test_helper(
            batch, batch_idx, "test"
        )

    def configure_optimizers(self):
        # Need to include the encoder and decoder parameters
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        return torch.optim.Adam(params, lr=self.lr)
    
# ----------------------------------------------------------------

class LatentDynamics(pl.LightningModule):
    def __init__(
        self, 
        fp_autoencoder_checkpoint=utils.FP_AUTOENCODER_CKPT, 
        lr=0.001
    ):
        super().__init__()
         
        # Save the inputs
        self.autoencoder = Autoencoder.load_from_checkpoint(fp_autoencoder_checkpoint)
        self.lr = lr
        
        # Convienent for determining device
        self._dummy_param = nn.Parameter(torch.empty(0))

        # Dynamics model can be fairly simple, takes in 
        # a latent state and an action and predicts the next
        # state (not latent)
        self.dim_state  = utils.DIM_STATE
        self.dim_action = utils.DIM_ACTION
        self.dim_latent = utils.DIM_LATENT
        self.dynamics = nn.Sequential(
            nn.Linear(self.dim_latent + self.dim_action, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.dim_latent),
        )

        # And we can encode/decode this using the autoencoder's weights
        # (frozen). Set to eval so that it isn't trained
        self.autoencoder.eval()      

        # Loss function is MSE
        self.loss_computer = nn.MSELoss()  
        
    def get_device(self):
        return self._dummy_param.device
    
    def get_z_next(self, z, a):
        # Predict the next latent state
        input_tensor = torch.cat(
            [
                z,
                a
            ], dim=1
        )
        z_next = self.dynamics(input_tensor)
        return z_next

    def forward(self, x):
        s, a, s_next = x
        z_next = self.get_z_next(self.autoencoder.encoder(s), a)
        # Decode this to get the next state
        s_next_hat = self.autoencoder.decoder(z_next)
        return s_next_hat, z_next

    def _train_val_test_helper(self, batch, batch_idx, name):
        """
        This function does a forward pass, computes the loss,
        and logs the loss (and other quantities of interest)
        """
        
        def simple_log(quantity_name, quantity):
            self.log(
                f"{name}_{quantity_name}", 
                quantity, 
                prog_bar=True,
                logger=True, 
                #on_step=True, 
                on_epoch=True,
            )

        # Extract inputs and prediction targets from batch
        s, a, s_next = batch
        s_next_hat, z_next = self.forward(batch)
        s_next_hat = s_next_hat.view_as(s_next)
        
        # Compute, log, and return the loss
        loss = self.loss_computer(s_next, s_next_hat)
        simple_log("loss (mse)", loss)
        return {
            "loss": loss
        }

    def training_step(self, batch, batch_idx):
        return self._train_val_test_helper(
            batch, batch_idx, "train"
        )

    def validation_step(self, batch, batch_idx):
        return self._train_val_test_helper(
            batch, batch_idx, "val"
        )

    def test_step(self, batch, batch_idx):        
        return self._train_val_test_helper(
            batch, batch_idx, "test"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.dynamics.parameters(), lr=self.lr)
    
# ----------------------------------------------------------------
    
class CompositeModel(pl.LightningModule):
    def __init__(self, autoencoder, dynamics):
        super().__init__()

        """
        This model is only ever used for inference, and it just takes
        a state and action and gives you the next state (no latent)
        """

        self.autoencoder = autoencoder
        self.dynamics = dynamics
        