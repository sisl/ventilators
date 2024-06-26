import csv
import os
import ast
from glob import glob
from typing import Any
from tqdm import tqdm
import numpy as np
import random
import warnings
import pandas as pd
import math

import pytorch_lightning as pl

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

"""
Need a couple data modules here, one for training the autoencoder
and one for training the dynamics model

The autoencoder is just states -> states, so the datamodule
should just include states data

The dynamics model is states & actions -> next states, so the
datamodule should include instances of s_n, a_n, s_n+1 
"""

def split_iterable_by_ratios(iterable, ratios):
    """
    Split an iterable into parts according to ratios
    """
    num_splits = len(ratios)
    num_items = len(iterable)
    split_indices = (np.cumsum(ratios) * num_items).astype(int)
    # Add the zero
    split_indices = np.insert(split_indices, 0, 0)
    # Could be a list so don't use np split 
    splits = []
    for i in range(num_splits):
        start = split_indices[i]
        end = split_indices[i+1]
        splits.append(iterable[start:end])
    return splits

def provide_consistent_dataloader(dataset, shuffle, batch_size, num_workers=4):
        """
        Return a dataloader from a dataset 
        """
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            # Using a high number here can make it take a long time to bring up initially
            # and is best avoided
            num_workers=num_workers,#os.cpu_count()-2, 
            persistent_workers=num_workers>0,
            shuffle=shuffle,
        )

def provide_consistent_dataset(data):
    # Provide a very simple dataset over some iterables, allowing
    # the data to be a list of objects
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, l):
            self.l = l
        def __len__(self):
            return len(self.l)
        def __getitem__(self, index):
            return self.l[index]
        def __reduce__(self):
            return (self.__class__, (self.l,))
    return ListDataset(data)

def instances_to_datasets(class_name, instances, verbose=True):
    # Use the ratios to split the data  
    instances_train, instances_val, instances_test = \
        split_iterable_by_ratios(instances, [0.7, 0.2, 0.1])
    
    if verbose:
        print(f"{class_name} data module with {len(instances)} instances")
        print(f"Train num: {len(instances_train)}")
        print(f"Val. num:  {len(instances_val)}")
        print(f"Test num:  {len(instances_test)}")

    # Now create the datasets, which return s, a, s' triplets
    ds_train = provide_consistent_dataset(instances_train)
    ds_val = provide_consistent_dataset(instances_val)
    ds_test = provide_consistent_dataset(instances_test)
    return ds_train, ds_val, ds_test

def return_fraction_of_instances(instances, fraction):
    # Random fraction, and not 1D so can't use np.random.choice
    num_instances = len(instances)
    num_instances_to_use = math.floor(fraction * num_instances)
    indices = random.sample(range(num_instances), num_instances_to_use)
    # Now get the instances, note that the iterable may be a list
    # or a numpy array
    instances_fraction = [instances[i] for i in indices]
    return instances_fraction

# ----------------------------------------------------------------

# When training, we change this to the actual data directories that
# we want to use
example_data_dirs = [
    "/mnt/learning/example_training_data/sim_2024-05-16-03-56-45/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-02-57/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-09-16/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-15-23/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-21-30/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-27-39/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-33-33/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-39-38/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-45-40/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-51-46/",
    "/mnt/learning/example_training_data/sim_2024-05-16-04-57-55/",
    "/mnt/learning/example_training_data/sim_2024-05-16-05-04-03/",
    "/mnt/learning/example_training_data/sim_2024-05-16-05-10-08/",
    "/mnt/learning/example_training_data/sim_2024-05-16-05-16-19/",
    "/mnt/learning/example_training_data/sim_2024-05-16-05-22-18/",
    "/mnt/learning/example_training_data/sim_2024-05-16-05-28-20/",
]

class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dirs:list=example_data_dirs, 
        batch_size:int=128,
        use_fraction_of_data:float=1.0,
    ):
        """
        This is for getting state data for the purpose of autoencoding,
        so we only need to load states
        """

        super().__init__()

        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.use_fraction_of_data = use_fraction_of_data

        # Now load all the data instances and normalize them
        instances = []
        for data_dir in tqdm(data_dirs, desc=f"Loading data from {len(self.data_dirs)} directories"):
            instances += self.instances_from_data_dir(data_dir)
        instances = np.array(instances)

        # Normalise the states
        self.states_mean = np.mean(instances, axis=0)
        self.states_std = np.std(instances, axis=0)
        instances = (instances - self.states_mean) / self.states_std
        
        # Split up into train, val, and test
        self.ds_train, self.ds_val, self.ds_test = instances_to_datasets(
            class_name=self.__class__.__name__,
            instances=instances,
        )

        # Report shapes
        print(f"States shape: {instances.shape}")

        #print(instances[0:2])

    def instances_from_data_dir(self, data_dir):
        # This is an autoencoder, so we just need to predict the states
        states_all = pd.read_csv(f"{data_dir}/states.csv")

        # Use a fraction of the data (shuffle sampling)
        instances = return_fraction_of_instances(states_all.values, self.use_fraction_of_data)
        
        # np arrays please, and float 32
        instances = np.array(instances)
        instances = torch.tensor(instances).float()

        return instances

    def setup(self, stage:str="", verbose=True):
        pass

    def train_dataloader(self):
        return provide_consistent_dataloader(self.ds_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return provide_consistent_dataloader(self.ds_train, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return provide_consistent_dataloader(self.ds_test, shuffle=False, batch_size=self.batch_size, num_workers=0)

# ----------------------------------------------------------------
    
class DynamicsDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dirs:list=example_data_dirs, 
        batch_size:int=32,
        use_fraction_of_data:float=1.0,
    ):
        """
        This is for predicting the next state given the current state given
        an action, in latent space. So we need to provide s, a, s' 
        triplets
        """

        super().__init__()

        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.use_fraction_of_data = use_fraction_of_data

        # Now load all the data instances and normalize them
        instances = []
        for data_dir in tqdm(data_dirs, desc=f"Loading data from {len(self.data_dirs)} directories"):
            instances += self.instances_from_data_dir(data_dir)

        # Pull the states and actions out of the instances
        states = np.array([x[0] for x in instances])
        actions = np.array([x[1] for x in instances])

        # Normalise the states
        self.states_mean = np.mean(states, axis=0)
        self.states_std = np.std(states, axis=0)
        self.actions_mean = np.mean(actions, axis=0)
        self.actions_std = np.std(actions, axis=0)
        def norm_triplet(triplet):
            s, a, s_prime = triplet
            s = (s - self.states_mean) / self.states_std
            a = (a - self.actions_mean) / self.actions_std
            s_prime = (s_prime - self.states_mean) / self.states_std
            return s, a, s_prime

        # Apply the computation
        instances = [norm_triplet(x) for x in instances]
        
        # Split up into train, val, and test
        self.ds_train, self.ds_val, self.ds_test = instances_to_datasets(
            class_name=self.__class__.__name__,
            instances=instances,
        )

        # Report shapes
        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")

    def instances_from_data_dir(self, data_dir):
        # Need s,a,s' triplets
        states_all = pd.read_csv(f"{data_dir}/states.csv")
        actions_all = pd.read_csv(f"{data_dir}/actions.csv")

        # Now we need to create the s, a, s' triplets
        sas_triplets = []
        for i in tqdm(range(len(states_all) - 1), desc="Sequencing states and actions into s, a, s' triplets"):
            s = states_all.iloc[i]
            a = actions_all.iloc[i]
            s_prime = states_all.iloc[i+1]
            # Convert to tuple of np arrays
            sas_triplet = (s.values, a.values, s_prime.values)
            # Convert to float 32
            sas_triplet = tuple([torch.tensor(x).float() for x in sas_triplet])
            sas_triplets.append(sas_triplet)

        # basically every instances is a tuple of 3 np arrays, and each np array 
        # is a state, action, or next state (so they have different dimensions)

        # Use a fraction of the data (shuffle sampling)
        instances = return_fraction_of_instances(sas_triplets, self.use_fraction_of_data)

        return instances

    def setup(self, stage:str="", verbose=True):
        pass

    def train_dataloader(self):
        return provide_consistent_dataloader(self.ds_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return provide_consistent_dataloader(self.ds_train, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return provide_consistent_dataloader(self.ds_test, shuffle=False, batch_size=self.batch_size, num_workers=0)

