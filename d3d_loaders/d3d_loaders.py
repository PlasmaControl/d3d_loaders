#!/usr/bin/env python

"""
Implements an iterable dataset for the HDF5 data stored in
/projects/EKOLEMEN/aza_lenny_data1
"""

import torch
from d3d_loaders.signal0d import *
from d3d_loaders.signal1d import *
from d3d_loaders.targets import *

import logging


class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, shotnr, 
                 predictors,
                 targets,
                 sampler_pred,
                 sampler_targ,
                 standardizer_dict,
                 datapath="/projects/EKOLEMEN/aza_lenny_data1",
                 device="cpu"):
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
            Time sampler for targets.

        standardizer_dict : dict {name : class `standardizer`}
            Provides individual standardizer for each predictor/target
                       
        device : string
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        Timebase of
        * ECE: dt_ece = 2e-3 ms
        * Pinj: dt_p = 1e-2 ms
        * Neutrons: dt_neu = 2e-2ms

        At each iteration, the data loader fetches a set of data for a given  shot for t0.
        In addition, the loader returns the target at a time t0 + shift_target

        For example, `t_params = {"tstart": 100.0, "tend": 2000.0, "tsample": 1.0}` will
        have all predictors signals span 100.0 - 2000.0 ms, with a sample length of 1.0ms.
        That is, the signals are 1900 elements each.
        The length of the target signals is the same. When shift_targets=None, the target
        signals will be identical to those defined in the predictor.
        When defining f.ex. `shift_targets = {"ae_prob": 10.0}`, the predictor signals `ae_prob`
        will be 10 samples (= 10.0ms / 1.0ms) ahead of the target signal.

        """

        super(D3D_dataset).__init__()
        self.datapath = datapath
        self.shotnr = shotnr
        self.sampler_pred = sampler_pred
        self.sampler_targ = sampler_targ
        self.standardizer_dict = standardizer_dict
        self.device = device
        logging.info(f"Using device {device}")

        self.predictors = {}
        self.targets = {}

        # Initialize all predictors
        logging.info("----Building predictor signals")
        for pred_name in predictors:
            logging.info(f"Adding {pred_name} to predictor list.")
            if pred_name == "pinj":
                self.predictors[pred_name] = signal_pinj(shotnr, self.sampler_pred, self.standardizer_dict[pred_name], self.datapath, device)
            elif pred_name == "pradcore":
                self.predictors[pred_name] = signal_pradcore(shotnr, self.sampler_pred, self.standardizer_dict[pred_name], self.datapath, device)
            elif pred_name == "pradedge":
                self.predictors[pred_name] = signal_pradedge(shotnr, self.sampler_pred, self.standardizer_dict[pred_name], self.datapath, device)
            else: 
                new_signal = signal_factory(f"signal_{pred_name}", signal_base)
                self.predictors[pred_name] = new_signal(shotnr, self.sampler_pred, self.standardizer_dict[pred_name], self.datapath, device)

            
        logging.info("----Building target signals")
        for target_name in targets:
            logging.info(f"Adding {target_name} to target list.")

            if target_name == "ae_prob_delta":
                self.targets[target_name] = signal_ae_prob_delta(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], datapath=self.datapath, device=device)
            elif target_name == "uci_label":
                self.targets[target_name] = signal_uci_label(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], datapath=self.datapath, device=device)
            elif target_name == "ae_prob":
                self.targets[target_name] = signal_ae_prob(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], datapath=self.datapath, device=device)
            elif target_name == "AE_predictions":
                self.targets[target_name] = signal_AE_pred(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], datapath=self.datapath, device=device)
            elif target_name == "betan":
                self.targets[target_name] = signal_betan(shotnr, self.sampler_targ, self.standardizer_dict[pred_name], datapath=self.datapath, device=device)
            # Add other targets here
            elif target_name == "ttd":
                self.targets[target_name] = target_ttd(shotnr, self.sampler_targ, datapath=self.datapath, device=device)
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

    Example use-case

    >>> shot_list_train = [172337, 172339] 
    >>> tstart = 110.0 # Time of first sample for upper triangularity is 100.0
    >>> tend = 2000.0
    >>> t_params = {"tstart": tstart, "tend": tend, "tsample": 1.0}
    >>> t_shift = 10.0
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> seq_length = 512
    >>> batch_size = 4
    >>> pred_list = ["pinj", "tinj", "ae_prob"]
    >>> targ_list = ["ae_prob"]
    >>> ds_train = Multishot_dataset(shot_list_train, t_params, pred_list, targ_list, {"ae_prob": t_shift}, "/projects/EKOLEMEN/d3dloader/test", device)
    >>> loader_train_b = torch.utils.data.DataLoader(ds_train, num_workers=0, 
    >>>                                              batch_sampler=RandomBatchSequenceSampler_multids(2, len(ds_train.datasets[0]), seq_length, batch_size),
    >>>                                              collate_fn = collate_fn_random_batch_seq_multi(batch_size))
    >>> for xb, yb in loader_train_b:
            print(xb.shape, yb.shape)
            break
    torch.Size([513, 4, 7]) torch.Size([513, 4, 5])

    
    Args:
        shotlist: list[int]
        t_params: dict
        predictors: list[string]
        targets: list[string]
        shift_targets: dict
        datapath: string
        device: torch.device
    """
    def __init__(self, shotlist, t_params, predictors, targets, shift_targets, datapath, device):
        # Create list of D3D_datasets
        self.datasets = [D3D_dataset(shotnr, t_params, predictors, targets, shift_targets, datapath, device) for shotnr in shotlist]
        
    def shot(self, idx: int):
        r"""Quick access to individual shot"""
        return self.datasets[idx]

    def __len__(self):
        return len(self.dataset)
        
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
