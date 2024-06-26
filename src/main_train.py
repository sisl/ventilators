import sys
import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from datetime import datetime
import torch

# Custom imports
sys.path.append('/mnt/src/')
import datreq
import plotting
import utils
from vents import Ventilator
import rewards
import patient
from patient import Patient
import agents
import environment
import learning.data_module
import learning.model

# Take advantage of tensor cores
# 'medium' | 'high'
torch.set_float32_matmul_precision('medium')

def init_new_wandb_run_and_get_logger(project_name, model_class):
    run, run_name = utils.init_new_wandb_run(project_name, model_class)
    return WandbLogger(
        project=project_name,
        name=run_name
    )

def get_consistent_trainer(num_epochs, name):
    return pl.Trainer(
        logger=init_new_wandb_run_and_get_logger("ventilators-training", name),
        max_epochs=num_epochs,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        default_root_dir="/mnt/learning/",
        # Number of devices, not index
        devices=1,
        accelerator="cpu",
        # We'll exceed the print rate of most notebooks if we don't use
        # This number is how many batches to update after
        callbacks=[TQDMProgressBar(refresh_rate=8)],
    )

def train_autoencoder():
    # Train the autoencoder
    dm = learning.data_module.AutoencoderDataModule(use_fraction_of_data=1.0)
    model = learning.model.Autoencoder(
        lr=0.005,
    )
    print("Training model:")
    print(model)
    trainer = get_consistent_trainer(num_epochs=256, name="autoencoder")
    trainer.fit(model, dm)
    fp_checkpoint = f"/mnt/learning/trained_models/autoencoder-{utils.get_timestamp()}.ckpt"
    trainer.save_checkpoint(fp_checkpoint)
    trainer.test(model, dm)
    return model, fp_checkpoint

def train_latent_dynamics(fp_autoencoder_checkpoint):
    # Train the dynamics model (using the frozen autoencoder)
    dm = learning.data_module.DynamicsDataModule(use_fraction_of_data=1.0)
    model = learning.model.LatentDynamics(
        fp_autoencoder_checkpoint=fp_autoencoder_checkpoint,
        lr=0.0005,
    )
    print("Training model:")
    print(model)
    trainer = get_consistent_trainer(num_epochs=256, name="dynamics")
    trainer.fit(model, dm)
    fp_checkpoint = f"/mnt/learning/trained_models/dynamics-{utils.get_timestamp()}.ckpt"
    trainer.save_checkpoint(fp_checkpoint)
    trainer.test(model, dm)
    return model, fp_checkpoint

# python3 /mnt/src/main_train.py
if __name__ == "__main__":
    # Here we train an autoencoder + latent dynamics
    # model from simulation data (histories)

    # Train/load the autoencoder?
    train = True 
    if train:
        autoencoder, fp_autoencoder_checkpoint = train_autoencoder()
    else:
        # This is a backup checkpoint that we know works
        fp_autoencoder_checkpoint = utils.FP_AUTOENCODER_CKPT

    # Train the dynamics model (using the frozen autoencoder)
    latent_dynamics, fp_latent_dynamics_checkpoint = train_latent_dynamics(fp_autoencoder_checkpoint)

    # Everything is now saved
    print(f"Autoencoder checkpoint: {fp_autoencoder_checkpoint}")
    print(f"Latent dynamics checkpoint: {fp_latent_dynamics_checkpoint}")