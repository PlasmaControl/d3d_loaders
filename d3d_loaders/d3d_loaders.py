#!/usr/bin/env python

"""
Implements an iterable dataset for the HDF5 data stored in
/projects/EKOLEMEN/aza_lenny_data1
"""

import torch
from d3d_loaders.signal1d import *
from d3d_loaders.signal2d import *

import logging


class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, shotnr, t_params, 
                 predictors,
                 targets,
                 shift_targets, 
                 datapath="/projects/EKOLEMEN/aza_lenny_data1",
                 device="cpu"):
        """Initializes the dataloader for DIII-D data.

        Parameters
        ----------
        shotnr : int 
                 shot number

        t_params : dict
                   Contains the following keys:
        
                        tstart : float
                                Start time, in milliseconds
                        tend : float
                            End time, in milliseconds
                        tsample : float
                                Time between samples, in milliseconds. Data is up or down sampled
                                to reach this time. Always uses most recent measurement, no averaging.
        shift_targets : dict
                       Defines an offset that is added for the target signal timebase.
                       shift_targets = {"ae_prob": 2.0} 
                       Keys have to match names of the predictors or targets.
                       Values define an offset that is added to the signals timebase
                       EXCEPT FOR: ae_prob_delta where it is the amount we look 
                       into the future to calculate our change in probability.
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
        self.tstart = t_params["tstart"]
        self.tend = t_params["tend"]
        self.tsample = t_params["tsample"]
        self.shift_targets = shift_targets
        self.device = device

        logging.info(f"Using device {device}")

        if self.tsample == -1:
            raise(ValueError('Cannot load full data with using full dataloader (tsample==-1)'))
        assert(self.tstart < self.tend)
        assert((self.tend - self.tstart) > self.tsample)
        self.predictors = {}
        self.targets = {}

        # Initialize all predictors
        logging.info("---Building predictor signals")
        for pred_name in predictors:
            # # Get t_shift from shift_target
            # t_params_key = t_params.copy()
            # try:
            #     t_shift = self.shift_target[pred_name]
            # except:
            #     t_shift = 0.0
            # t_params_key["t_shift"] = t_shift
            # Load Signal
            logging.info(f"Adding {pred_name} to predictor list.")
            if pred_name == "pinj":
                self.predictors[pred_name] = signal_pinj(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "tinj":
                self.predictors[pred_name] = signal_tinj(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "bmspinj":
                self.predictors[pred_name] = signal_bmspinj(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "bmstinj":
                self.predictors[pred_name] = signal_bmstinj(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "ae_prob":
                self.predictors[pred_name] = signal_ae_prob(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "neutronsrate":
                self.predictors[pred_name] = signal_neut(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "ip":
                self.predictors[pred_name] = signal_ip(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "iptipp":
                self.predictors[pred_name] = signal_iptipp(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "dstdenp":
                self.predictors[pred_name] = signal_dstdenp(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "dssdenest":
                self.predictors[pred_name] = signal_dssdenest(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "echpwrc":
                self.predictors[pred_name] = signal_ech(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "ali":
                self.predictors[pred_name] = signal_ech(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "q95":
                self.predictors[pred_name] = signal_q95(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "kappa":
                self.predictors[pred_name] = signal_kappa(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "pcbcoil":
                self.predictors[pred_name] = signal_pcbcoil(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "dens":
                self.predictors[pred_name] = signal_dens(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "pres":
                self.predictors[pred_name] = signal_pres(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "temp":
                self.predictors[pred_name] = signal_temp(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "q":
                self.predictors[pred_name] = signal_q(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "doutl":
                self.predictors[pred_name] = signal_doutl(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "doutu":
                self.predictors[pred_name] = signal_doutu(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "tritop":
                self.predictors[pred_name] = signal_tritop(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "tribot":
                self.predictors[pred_name] = signal_tritop(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "raw_ece":
                # Currently no way to subset ECE channels
                # channels = []
                self.predictors[pred_name] = signal_ece(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "raw_co2_dp":
                # Currently no way to subset CO2 channels
                # channels = []
                self.predictors[pred_name] = signal_co2_dp(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "raw_co2_pl":
                # Currently no way to subset CO2 channels
                # channels = []
                self.predictors[pred_name] = signal_co2_pl(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "raw_mpi":
                # Currently no way to subset MPI angles
                # angles = []
                self.predictors[pred_name] = signal_mpi(shotnr, t_params, datapath=self.datapath, device=device)
            elif pred_name == "raw_bes":
                # Currently no way to subset BES channels
                # channels = []
                self.predictors[pred_name] = signal_BES(shotnr, t_params, datapath=self.datapath, device=device)
            
            # Add other predictors here
            else:
                raise(ValueError(f'{pred_name} is not a valid predictor'))
            
        logging.info("----Building target signals")
        # Check that target has a time_shift defined.
        for target_name in targets:
            if target_name not in shift_targets.keys():
                raise KeyError(f"{target_name} is specified as a target, but time shift is not defined in shift_targets")

        for target_name in targets:
            t_params_target = t_params.copy()
            t_params_target["tshift"] = shift_targets[target_name]

            logging.info(f"Adding {target_name} to target list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_params_target['tshift']}ms")


            if target_name == "ae_prob_delta":
                self.targets[target_name] = signal_ae_prob_delta(shotnr, t_params_target, datapath=self.datapath, device=device)
            elif target_name == "uci_label":
                self.targets[target_name] = signal_uci_label(shotnr, t_params_target, datapath=self.datapath, device=device)
            elif target_name == "ae_prob":
                self.targets[target_name] = signal_ae_prob(shotnr, t_params_target, datapath=self.datapath, device=device)
            elif target_name == "AE_predictions":
                self.targets[target_name] = signal_AE_pred(shotnr, t_params_target, datapath=self.datapath, device=device)
            elif target_name == "betan":
                self.targets[target_name] = signal_betan(shotnr, t_params_target, datapath=self.datapath, device=device)
            # Add other targets here
            else:
                raise(ValueError(f'{target_name} is not a valid target'))


        # Assert that all data has the same number of samples
        base_key = next(iter(self.predictors.keys()))
        for k in self.predictors.keys():
            assert(self.predictors[k].data.shape[0] == self.predictors[base_key].data.shape[0])

        for k in self.targets.keys():
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
