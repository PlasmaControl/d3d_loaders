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
            if pred_name == "pinj":
                logging.info(f"Adding pinj to predictor list.")
                self.predictors["pinj"] = signal_pinj(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "tinj":
                logging.info(f"Adding pinj to predictor list.")
                self.predictors["tinj"] = signal_pinj(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "ae_prob":
                logging.info(f"Adding ae_prob to predictor list.")
                self.predictors["ae_prob"] = signal_ae_prob(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "neutronsrate":
                logging.info(f"Adding neutron rate to predictor list.")         
                self.predictors["neut"] = signal_neut(shotnr, t_params, datapath=self.datapath, device=device)
            
            elif pred_name == "ip":
                logging.info(f"Adding injected power to predictor list.")         
                self.predictors["ip"] = signal_ip(shotnr, t_params, datapath=self.datapath, device=device)
            
            elif pred_name == "iptipp":
                logging.info(f"Adding target current to predictor list.")
                self.predictors["iptipp"] = signal_iptipp(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "dstdenp":
                logging.info(f"Adding target density to predictor list.")
                self.predictors["dstdenp"] = signal_dstdenp(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "dssdenest":
                logging.info("Adding line-averaged density to predictor list")
                self.predictors["dssdenest"] = signal_dssdenest(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "echpwrc":
                logging.info(f"Adding ECH to predictor list.")         
                self.predictors["echpwrc"] = signal_ech(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "ali":
                logging.info(f"Adding Internal inductance to predictor list.")         
                self.predictors["ali"] = signal_ech(shotnr, t_params, datapath=self.datapath, device=device)

            elif pred_name == "q95":
                logging.info(f"Adding q95 to predictor list.")         
                self.predictors["q95"] = signal_q95(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "kappa":
                logging.info(f"Adding kappa to predictor list.")         
                self.predictors["kappa"] = signal_kappa(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "dens":
                logging.info(f"Adding density profile to predictor list.")         
                self.predictors["dens"] = signal_dens(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "pres":
                logging.info(f"Adding pressure profile to predictor list.")         
                self.predictors["pres"] = signal_pres(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "temp":
                logging.info(f"Adding temperature profile to predictor list.")         
                self.predictors["temp"] = signal_temp(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "q":
                logging.info(f"Adding q profile to predictor list.")         
                self.predictors["q"] = signal_q(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "doutl":
                logging.info(f"Adding lower triangularity to predictor list.")         
                self.predictors["tri_l"] = signal_doutl(shotnr, t_params, datapath=self.datapath, device=device)
            
            elif pred_name == "doutu":
                logging.info(f"Adding upper triangularity to predictor list.")         
                self.predictors["tri_u"] = signal_doutu(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "raw_ece":
                logging.info(f"Adding raw ECE signals to predictor list.") 
                # Currently no way to subset ECE channels
                # channels = []
                self.predictors["raw_ece"] = signal_ece(shotnr, t_params, datapath=self.datapath, device=device)
            
            elif pred_name == "raw_co2_dp":
                logging.info(f"Adding raw CO2 dp signals to predictor list.") 
                # Currently no way to subset CO2 channels
                # channels = []
                self.predictors["raw_co2_dp"] = signal_co2_dp(shotnr, t_params, datapath=self.datapath, device=device)
                
            elif pred_name == "raw_co2_pl":
                logging.info(f"Adding raw CO2 pl signals to predictor list.") 
                # Currently no way to subset CO2 channels
                # channels = []
                self.predictors["raw_co2_pl"] = signal_co2_pl(shotnr, t_params, datapath=self.datapath, device=device)
            
            elif pred_name == "raw_mpi":
                logging.info(f"Adding raw MPI dp signals to predictor list.") 
                # Currently no way to subset MPI angles
                # angles = []
                self.predictors["raw_mpi"] = signal_mpi(shotnr, t_params, datapath=self.datapath, device=device)
            
            elif pred_name == "raw_bes":
                logging.info(f"Adding raw BES signals to predictor list.") 
                # Currently no way to subset BES channels
                # channels = []
                self.predictors["raw_bes"] = signal_BES(shotnr, t_params, datapath=self.datapath, device=device)
            
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
            
            if target_name == "ae_prob_delta":
                logging.info(f"Adding ae_prob_delta to target list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_params_target['tshift']}ms")
                self.targets["ae_prob_delta"] = signal_ae_prob_delta(shotnr, t_params_target, datapath=self.datapath, device=device)
            
            elif target_name == "uci_label":
                logging.info(f"Adding uci_label to target list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_params_target['tshift']}ms")
                self.targets["uci_label"] = signal_uci_label(shotnr, t_params_target, datapath=self.datapath, device=device)

            elif target_name == "ae_prob":
                logging.info(f"Adding ae_prob_delta to target list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_params_target['tshift']}ms")
                self.targets["ae_prob"] = signal_ae_prob(shotnr, t_params_target, datapath=self.datapath, device=device)
 
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
