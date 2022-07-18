#!/usr/bin/env python

"""
Implements an iterable dataset for the HDF5 data stored in
/projects/EKOLEMEN/aza_lenny_data1
"""

import torch
from d3d_loaders.signal1d import signal_pinj, signal_neut
from d3d_loaders.signal2d import signal_ae_prob, signal_ae_prob_delta

import logging



class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, shotnr, t_params,
                 shift_targets={'ae_prob_delta':10.0}, 
                 predictors=["pinj", "neutrons", "q95_prof"],
                 targets=["ae_prob_delta"],
                 datapath="/projects/EKOLEMEN/aza_lenny_data1",
                 device="cpu"):
        """Initializes the dataloader for DIII-D data.

        Parameters
        ----------
        shotnr : array of ints
                 shot numbers

        t_params : dict
                   Contains the following keys:
        
                        tstart : float
                                Start time, in milliseconds
                        tend : float
                            End time, in milliseconds
                        tsample : float
                                Time between samples, in milliseconds
        shift_targets : dict
                       Keys have to match names of the predictors or targets.
                       Values define an offset that is added to the signals timebase.
        device : string
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        Timebase of
        * ECE: dt_ece = 2e-3 ms
        * Pinj: dt_p = 1e-2 ms
        * Neutrons: dt_neu = 2e-2ms

        At each iteration, the data loader fetches a set of data for a given 
        shot for t0.
        In addition, the loader returns the target at a time t0 + shift_target

        Data is sampled at tsample. tsample should be an integer multiple of
        dt_ece, dt_p, and dt_neu.

        """

        super(D3D_dataset).__init__()
        self.datapath = datapath
        self.shotnr = shotnr
        self.tstart = t_params["tstart"]
        self.tend = t_params["tend"]
        self.tsample = t_params["tsample"]
        self.shift_targets = shift_targets

        logging.info(f"Using device {device}")

        assert(self.tstart < self.tend)
        assert((self.tend - self.tstart) > self.tsample)
        self.predictors = {}
        self.targets = {}

        # Initialize all predictors
        for pred_name in predictors:
            # Get t_shift from shift_target
            t_params_key = t_params.copy()
            try:
                t_shift = self.shift_target[pred_name]
            except:
                t_shift = 0.0
            t_params_key["t_shift"] = t_shift
            
            if pred_name == "pinj":
                logging.info(f"Adding pinj to predictor list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_shift}")
                self.predictors["pinj"] = signal_pinj(shotnr, t_params_key, device=device)
                
            if pred_name == "ae_prob":
                logging.info(f"Adding ae_prob to predictor list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_shift}")
                self.predictors["ae_prob"] = signal_ae_prob(shotnr, t_params_key, device=device)
                
            if pred_name == "neut":
                logging.info(f"Adding neutron rate to predictor list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_shift}")         
                self.predictors["neut"] = signal_neut(shotnr, t_params_key, device=device)
                
            # Add other predictors here


        for target_name in targets:
            t_params_key = t_params.copy()
            try:
                t_shift = self.shift_target[target_name]
            except:
                t_shift = 0.0 
            t_params_key['t_shift'] = t_shift
            
            if target_name == "ae_prob_delta":
                logging.info(f"Adding ae_prob_delta to target list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_shift}")              
                self.targets["ae_prob_delta"] = signal_ae_prob_delta(shotnr, t_params_key,
                                                                     device=device)
            
            # Add other targets here
            

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
#
# End of file d3d_loaders.py
