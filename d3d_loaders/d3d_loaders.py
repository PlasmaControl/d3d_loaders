#!/usr/bin/env python

"""
Implements an iterable dataset for D3D data stored in HDF5 format.
"""
from os.path import join
import importlib.resources
import logging
import torch
import yaml


import d3d_signals

from d3d_loaders.signal0d import signal_base, signal_pinj, signal_factory
from d3d_loaders.signal1d import signal_1d_base, profile_factory
from d3d_loaders.targets import target_ttd, target_ttelm


class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, shotnr, 
                 predictors,
                 targets,
                 sampler_pred,
                 sampler_targ,
                 ip_space,
                 standardizer_dict,
                 datapath,
                 device=torch.device("cpu")):
        """Initializes the dataloader for DIII-D data.

        Parameters
        ----------
        shotnr : int 
                 shot number

        predictors : list(str)

        targets : list(str)

        sampler_pred : class `sampler_base`
            Provides a uniform time sampler for each predictor

        sampler_targ : class `sampler_base`
            Time sampler for targets. May be different to allow time shifting etc.

        ip_space : class `sampler_space`
            Interpolator for profiles

        standardizer_dict : dict {name : class `standardizer`}
            Provides individual standardizer for each predictor/target
                       
        device : string
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        super(D3D_dataset).__init__()
        self.datapath = datapath
        self.shotnr = shotnr
        self.sampler_pred = sampler_pred
        self.sampler_targ = sampler_targ
        self.ip_space = ip_space
        self.standardizer_dict = standardizer_dict
        self.device = device
        logging.info(f"Using device {device}")

        self.predictors = {}
        self.targets = {}

        # Load signal definitions
        resource_path = importlib.resources.files("d3d_signals")
        with open(join(resource_path, "signals_0d.yaml"), "r") as fp:
            self.signals_0d = yaml.safe_load(fp)

        with open(join(resource_path, "signals_1d.yaml"), "r") as fp:
            self.signals_1d = yaml.safe_load(fp)


        # Initialize all predictors. These can be either signals with special constructor, scalar or profile time series
        names_scalars = list(self.signals_0d.keys())
        names_profiles = list(self.signals_1d.keys())
        logging.info("----Building predictor signals")
        for pred_name in predictors:
            logging.info(f"Adding {pred_name} to predictor list.")
            # Find out wh

            # There are some signals that have a custom constructor.
            if pred_name == "pinj":
                self.predictors[pred_name] = signal_pinj(shotnr, self.sampler_pred, self.standardizer_dict[pred_name], self.datapath, self.device)
                continue

            # For all other signals, see if we need to instantiate a scalar or profile
            if pred_name in names_scalars: 
                new_signal = signal_factory(f"signal_{pred_name}")
                self.predictors[pred_name] = new_signal(shotnr, self.sampler_pred, self.standardizer_dict[pred_name], self.datapath, self.device)
            elif pred_name in names_profiles:
                new_signal = profile_factory(f"signal_{pred_name}")
                self.predictors[pred_name] = new_signal(shotnr, self.sampler_pred, self.ip_space, self.standardizer_dict[pred_name], self.datapath, self.device)
            else:
                logging.error(f"Signal for predictor {pred_name} not found")

                print(f"Signal for predictor {pred_name} not found")

            
        logging.info("----Building target signals")
        for target_name in targets:
            logging.info(f"Adding {target_name} to target list.")

            if target_name == "ae_prob_delta":
                self.targets[target_name] = signal_ae_prob_delta(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], self.datapath, self.device)
            elif target_name == "uci_label":
                self.targets[target_name] = signal_uci_label(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], self.datapath, self.device)
            elif target_name == "ae_prob":
                self.targets[target_name] = signal_ae_prob(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], self.datapath, self.device)
            elif target_name == "AE_predictions":
                self.targets[target_name] = signal_AE_pred(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], self.datapath, self.device)
            elif target_name == "betan":
                self.targets[target_name] = signal_betan(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], self.datapath, self.device)
            # Add other targets here
            elif target_name == "ttd":
                self.targets[target_name] = target_ttd(shotnr, self.sampler_targ, self.datapath, self.device)
            elif target_name == "ttelm":
                self.targets[target_name] = target_ttelm(shotnr, self.sampler_targ, self.datapath, self.device)
            else:
                raise(ValueError(f'{target_name} is not a valid target'))


        # Assert that all data has the same number of samples
        base_key = next(iter(self.predictors.keys()))
        for k in self.predictors.keys():
            assert(self.predictors[k].data.shape[0] == self.predictors[base_key].data.shape[0])

        for k in self.targets.keys():
            #print(self.targets[k].data.shape[0], self.predictors[base_key].data.shape[0])
            assert(self.targets[k].data.shape[0] == self.predictors[base_key].data.shape[0])


    def __len__(self):
        """Returns the number of time samples."""
        k = next(iter(self.predictors.keys()))
        return self.predictors[k].data.shape[0]

   
    def __getitem__(self, idx):
        """Fetch data corresponding to the idx'th sample.

        Use RandomBatchSequenceSampler in combination with torch.DataLoader for sampling from this dataset.
        
        Returns
        -------
        data_t0 - tensor 
                  Concatenated predictors. dim0: sample, dim1: feature
        data_t1 - tensor
                  Concatenated targets. dim0: sample, dim1: feature

        """
        data_t0 = torch.cat([v.data[idx, :] for v in self.predictors.values()], dim=-1)
        data_t1 = torch.cat([t.data[idx, :] for t in self.targets.values()], dim=-1)

        return data_t0, data_t1


class Multishot_dataset():
    r"""Multishot dataset is a wrapper for a dataset consisting of multiple shots.
    
    It's basically a list of D3D_datasets. But the __getindex__ function is smart,
    mapping a sequential index onto individual member datasets.
   
    Args:
        shotlist : list[int]
            List of shots in the dataset
        predictors : list[string]
            List of predictors, see yaml files in d3d_signals
        targets : list[string]
            List of targets. Needs to be a group in the HDF5 files
        sampler_pred : class `sampler_base`
            Provides a uniform time sampler for each predictor
        sampler_targ : class `sampler_base`
            Time sampler for targets. May be different to allow time shifting etc.
        ip_space : class `sampler_space`
            Interpolator for profiles
        standardizer_dict : dict {name : class `standardizer`}
            Provides individual standardizer for each predictor/target
        datapath: string
            Root path of the dataset
        device: torch.device
            Optional. Device to store the data tensors on. Either CPU or GPU.
    """
    def __init__(self, shotlist, predictors, targets, sampler_pred_dict, sampler_targ_dict, ip_space, std_dict, datapath, device=torch.device("cpu")):
        # Create list of D3D_datasets
        self.datasets = []
        for shotnr in shotlist:
            self.datasets.append(D3D_dataset(shotnr, predictors, targets, sampler_pred_dict[shotnr], 
                                 sampler_targ_dict[shotnr], ip_space, std_dict, datapath, device))

    def shot(self, idx: int):
        r"""Quick access to individual shot"""
        return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)
        
    def __getitem__(self, idx):
        """Fetch data from random dataset.
        
        This method uses tuple indices, idx=(ix_ds, ix_sample), where `ix_ds` is an int that specifies the shot
        and `ix_sample` denotes the samples for the specified shot. Each shot has zero-based indices.
        
        Args:
            idx: Either tuple(int, range), or list[tuple(int, range)]. int specifies the shot and range the
                 idx passed to D3D_dataset.__getitem__.
        """
        if isinstance(idx, list):
            return [self.datasets[i[0]][i[1]] for i in idx]
        elif isinstance(idx, tuple):
            return self.datasets[idx[0]][idx[1]]

# End of file d3d_loaders.py
