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
                 predictors=["pinj", "neutrons", "q95_prof"],
                 targets=["ae_prob_delta"],
                 shift_targets={'ae_prob_delta':10.0}, 
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
                       Values define an offset that is added to the signals timebase, except for
                       ae_prob_delta where it is the amount we look into the future to calculate
                       our change in probability.
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
        logging.info(f"t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={shift_targets['ae_prob_delta']}")         
        for pred_name in predictors:
            # Get t_shift from shift_target
            t_params_key = t_params.copy()
            try:
                t_shift = self.shift_target[pred_name]
            except:
                t_shift = 0.0
            t_params_key["t_shift"] = t_shift
            
            if pred_name == "pinj":
                logging.info(f"Adding pinj to predictor list.")
                self.predictors["pinj"] = signal_pinj(shotnr, t_params_key, device=device)
                
            elif pred_name == "ae_prob":
                logging.info(f"Adding ae_prob to predictor list.")
                self.predictors["ae_prob"] = signal_ae_prob(shotnr, t_params_key, device=device)
                
            elif pred_name == "neut":
                logging.info(f"Adding neutron rate to predictor list.")         
                self.predictors["neut"] = signal_neut(shotnr, t_params_key, device=device)
            
            elif pred_name == "ip":
                logging.info(f"Adding injected power to predictor list.")         
                self.predictors["ip"] = signal_ip(shotnr, t_params_key, device=device)
                
            elif pred_name == "ech":
                logging.info(f"Adding ECH to predictor list.")         
                self.predictors["ech"] = signal_ech(shotnr, t_params_key, device=device)
            
            elif pred_name == "q95":
                logging.info(f"Adding q95 to predictor list.")         
                self.predictors["q95"] = signal_q95(shotnr, t_params_key, device=device)
                
            elif pred_name == "kappa":
                logging.info(f"Adding kappa to predictor list.")         
                self.predictors["kappa"] = signal_kappa(shotnr, t_params_key, device=device)
                
            elif pred_name == "dens":
                logging.info(f"Adding density profile to predictor list.")         
                self.predictors["dens"] = signal_dens(shotnr, t_params_key, device=device)
                
            elif pred_name == "pres":
                logging.info(f"Adding pressure profile to predictor list.")         
                self.predictors["pres"] = signal_pres(shotnr, t_params_key, device=device)
                
            elif pred_name == "temp":
                logging.info(f"Adding temperature profile to predictor list.")         
                self.predictors["temp"] = signal_temp(shotnr, t_params_key, device=device)
                
            elif pred_name == "q":
                logging.info(f"Adding q profile to predictor list.")         
                self.predictors["q"] = signal_q(shotnr, t_params_key, device=device)
                
            elif pred_name == "tri_l":
                logging.info(f"Adding lower triangularity to predictor list.")         
                self.predictors["tri_l"] = signal_tri_l(shotnr, t_params_key, device=device)
            
            elif pred_name == "tri_u":
                logging.info(f"Adding upper triangularity to predictor list.")         
                self.predictors["tri_u"] = signal_tri_u(shotnr, t_params_key, device=device)
                
            elif pred_name == "raw_ece":
                logging.info(f"Adding raw ECE signals to predictor list.") 
                channels = []
                self.predictors["raw_ece"] = signal_ece(shotnr, t_params_key, device=device, channels=channels)
            
            else:
                raise(ValueError(f'{pred_name} is not a valid predictor'))
            # Add other predictors here


        for target_name in targets:
            t_params_key = t_params.copy()
            try:
                t_shift = self.shift_target[target_name]
            except:
                t_shift = 10.0 
            
            if target_name == "ae_prob_delta":
                logging.info(f"Adding ae_prob_delta to target list: t = {self.tstart}-{self.tend}ms, tsample={self.tsample}ms, t_shift={t_shift}")              
                self.targets["ae_prob_delta"] = signal_ae_prob_delta(shotnr, t_params_key, t_shift, device=device)
            
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
