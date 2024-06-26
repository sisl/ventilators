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
import learning.model
import utils

"""
Provide a utility function/class that can be used to do rollouts
of the dynamics model to approximate some future state
"""

class DynamicsPredictor:
    def __init__(
        self,
        fp_autoencoder_checkpoint=utils.FP_AUTOENCODER_CKPT, 
        fp_latent_dynamics_checkpoint=utils.FP_LATENT_DYNAMICS_CKPT,
    ):
        # Load the autoencoder and dynamics model
        self.autoencoder = learning.model.Autoencoder.load_from_checkpoint(fp_autoencoder_checkpoint)
        self.latent_dynamics = learning.model.LatentDynamics.load_from_checkpoint(fp_latent_dynamics_checkpoint)

        # All in eval mode
        self.autoencoder.eval()
        self.latent_dynamics.eval()

        # Need the latent dynamics data module because
        # we work in normalised action and state space
        self.dm = learning.data_module.DynamicsDataModule(use_fraction_of_data=1.0)

    def normalize_state(self, state):
        return (state - self.dm.states_mean) / self.dm.states_std
    def denormalize_state(self, state):
        return state * self.dm.states_std + self.dm.states_mean
    def normalize_action(self, action):
        return (action - self.dm.actions_mean) / self.dm.actions_std
    def denormalize_action(self, action):
        return action * self.dm.actions_std + self.dm.actions_mean

    def next_state(self, state, action):
        """
        Given a state and an action, predict the next state
        """

        # Make states and actions into lists
        state = list(state)
        action = list(action)

        # Normalise the state and action
        state_norm = self.normalize_state(state)
        action_norm = self.normalize_action(action)

        # Get the compressed representation of the next state,
        # note that we don't know what the next state is. Note
        # that this must be unwrapped in the forward call
        # of the dynamics model. And each element must be a float
        # 32 tensor
        sas_triplet = (
            torch.tensor(state_norm).float().unsqueeze(0),
            torch.tensor(action_norm).float().unsqueeze(0),
            # Don't actually know what the next state is
            None
        )
        s_next, z_next = self.latent_dynamics(sas_triplet)

        # The outputs are tensors 
        s_next = s_next.squeeze(0).detach().numpy()
        z_next = z_next.squeeze(0).detach().numpy()

        # And denormalise the state
        return self.denormalize_state(s_next)