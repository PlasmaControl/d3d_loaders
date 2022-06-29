#!/usr/bin/env python

"""
Implements an iterable dataset for the HDF5 data stored in
/projects/EKOLEMEN/aza_lenny_data1
"""

from os.path import join
import math
import h5py
import numpy as np
import torch
import time
from math import ceil

import pickle
from d3d_loaders.rcn_functions import rcn_infer

import logging



class signal_1d():
    """Base class for a 1d sample.

    Represents a 1-d signal over [tstart:tend], sampled with tsample.

    


    Setters and getters are passed to self.data tensor.
    """
    def __init__(self, shotnr, tstart, tend, tsample, 
            tshift=0.0, override_dt=None, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Load data from HDF5 file, standardize, and move to device.

        Input:
        ===== =
        shotnr......: (Int) Shot number
        tstart......: (float) Start of signal interval, in milliseconds
        tend........: (float) End of signal interval, in milliseconds
        tsample.....: (float) Desired sampling time, in milliseconds
        tshift......: (float) Shift signal by tshift with respect to tstart, in milliseconds
        override_dt.: (float) Use this value as sample spacing instead of calculating from xdata field in HDF5 file
        datapath....: (string) Basepath where HDF5 data is stored
        device......: (string) device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        """
        # Store function arguments as member variables
        self.shotnr = shotnr
        self.tstart = tstart
        self.tend = tend
        self.tsample = tsample
        self.tshift = tshift
        self.override_dt = override_dt
        self.datapath = datapath
        
        print("signal_1d, device=", device)

        # Load data from HDF5 file and store, move to device
        self.data = self._cache_data().to(device)
        # Z-score normalization
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data = (self.data - self.data_mean) / self.data_std
        logging.info(f"Compiled signald data for shot {shotnr}, mean={self.data_mean}, std={self.data_std}")


    def _get_num_n_samples(self, dt):
        """Calculate number of samples and sample skipping.

        Given a signal sampled on [tstart:tend] with sample spacing dt
        calculate the total number of samples available.

        Also calculates the sample skipping when sub-sampling dt to self.tsample.
        """

        num_samples = int(ceil((self.tend - self.tstart) / dt))
        nth_sample = int(ceil(self.tsample / dt))

        return (num_samples, nth_sample)

    def _cache_data(self):
        """Load data from HDF5. To be overriden by derived classes."""
        pass

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"signal_1d({self.data}"


class signal_pinj(signal_1d):
    """Sum of total injected power."""
    def _cache_data(self):
        """Loads sum of all pinj"""

        # Calculate samples for pinj data
        # Load pinj data at t0 and t0 + 50ms. dt for this data is 10ms
        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader
        fp_pinj = h5py.File(join(self.datapath, "template", f"{self.shotnr}_pinj.h5")) 
        tb_pinj = torch.tensor(fp_pinj["pinjf_15l"]["xdata"][:]) # Get time-base
        if self.override_dt is None:
            self.dt = (tb_pinji[1:] - tb_pinj[:-1]).mean()           # Get sampling time
        else:
            self.dt = self.override_dt
        # Get total number of samples and desired sub-sample spacing
        num_samples, nth_sample = self._get_num_n_samples(self.dt)
        t0_idx = torch.argmin(torch.abs(tb_pinj - self.tstart))
        logging.info(f"Sampling pinj: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        pinj_data = sum([torch.tensor(fp_pinj[k]["zdata"][:])[t0_idx:t0_idx + num_samples:nth_sample] for k in fp_pinj.keys()])
        fp_pinj.close()

        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading pinj for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
           
        # Return data with singular feature dimension
        # dim0: feature 
        # dim1: sample
        return pinj_data.unsqueeze(1)


class signal_neut(signal_1d):
    def _cache_data(self):
        """Loads neutron emission rate"""
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        # Don't use with... scope. This throws off dataloader
        fp_prof = h5py.File(join(self.datapath, "template", f"{self.shotnr}_profiles.h5")) 
    
        tb_neu = torch.tensor(fp_prof["neutronsrate"]["xdata"][:])
        if self.override_dt is None:
            self.dt = np.diff(tb_neu).mean()
        else:
            self.dt = self.override_dt

        num_samples, nth_sample = self._get_num_n_samples(self.dt)
        t0_idx = torch.argmin(torch.abs(tb_neu - self.tstart))
        logging.info(f"Sampling neut: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        neutron_data = torch.tensor(fp_prof["neutronsrate"]["zdata"][t0_idx:t0_idx + num_samples:nth_sample])
        fp_prof.close()    
        elapsed = time.time() - t0_p
       
        logging.info(f"Loading neutron data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")

        # Return data with singular feature dimension
        # dim0: feature 
        # dim1: sample
        return neutron_data.unsqueeze(1)
   

class signal_ae_prob(signal_1d):
    """Probability for a given Alfven Eigenmode."""
    def __init__(self, shotnr, tstart, tend, tsample, 
            tshift=0.0, override_dt=None, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Loads weights for RCN model and calls base class constructor."""
        # The RCN model weights are needed to call the base-class constructor
        with open('/home/rkube/ml4control/1dsignal-model-AE-ECE-RCN.pkl', 'rb') as df:
            self.infer_data = pickle.load(df)
        self.n_res_l1 = self.infer_data['layer1']['w_in'].shape[0]
        self.n_res_l2 = self.infer_data['layer2']['w_in'].shape[0]

        # Call base class constructor to fetch and store data
        signal_1d.__init__(self, shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device=device)


    def _cache_data(self):
        """Loads ECE data and calculates AE probability using Aza's RCN model.

        This has to be done sequential. Iterative over all samples."""

        # Find how many samples apart tsample is
        t0_p = time.time()
        # Don't use scope. This throws off multi-threaded loaders
        fp_ece = h5py.File(join(self.datapath, "template", f"{self.shotnr}_ece.h5"), "r") 
        tb_ece = fp_ece["ece"]["xdata"][:]    # Get ECE time-base
        dt_ece = np.diff(tb_ece).mean()   # Get ECE sampling time
        num_samples, nth_sample = self._get_num_n_samples(dt_ece)
        shift_smp = int(ceil(self.tshift/ dt_ece))
        t0_idx = np.argmin(np.abs(tb_ece - self.tstart))
        logging.info(f"Sampling ECE: t0_idx={t0_idx}, dt={dt_ece}, num_samples={num_samples}, nth_sample={nth_sample}, shift_smp={shift_smp}")

        # Read in all ece_data at t0 and shifted at t0 + 50 mus
        ece_data = np.vstack([fp_ece["ece"][f"tecef{(i+1):02d}"][t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample ] for i in range(40)]).T
        fp_ece.close()
        # After this we have ece_data_0.shape = (num_samples / nth_sample, 40)

        # Pre-allocate array for AE mode probabilities
        # dim0: time index
        # dim1: AE mode index 0...4
        ae_probs = np.zeros([ece_data.shape[0], 5], dtype=np.float32)

        ece_data = (ece_data - self.infer_data["mean"]) / self.infer_data["std"]
        # Initialize to zero, overwrite in for loop
        r_prev = {"layer1": np.zeros(self.n_res_l1),
                  "layer2": np.zeros(self.n_res_l2)} 
        # Iterate over time index 0
        for idx, u in enumerate(ece_data):
            L = "layer1"
            r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                     self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                     self.infer_data[L]['leak_rate'], r_prev[L], u.T)

            L = "layer2"
            r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                     self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                     self.infer_data[L]['leak_rate'], r_prev[L], y)
            ae_probs[idx, :] = y[:]

        elapsed = time.time() - t0_p

        logging.info(f"AE forward model for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")

        # Convert to torch tensor. dim0: feature. dim1: sample index
        return torch.tensor(ae_probs)


class signal_ae_prob_delta(signal_1d):
    """Change in Alfven Eigenmode probability over time""" 
    def __init__(self, shotnr, tstart, tend, tsample, 
            tshift=0.0, override_dt=None, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Construct difference in AE probability using two signal_ae_prob"""

        # Signal at t0
        signal_t0 = signal_ae_prob(shotnr, tstart, tend, tsample, tshift=0.0, 
                override_dt=override_dt, datapath=datapath, device=device)
        # Shifted signal
        signal_t1 = signal_ae_prob(shotnr, tstart, tend, tsample, tshift, 
                override_dt, datapath, device=device)
    
        self.shotnr = shotnr
        self.tstart = tstart
        self.tend = tend
        self.tsample = tsample
        self.tshift = tshift
        self.override_dt = override_dt
        self.datapath = datapath

        self.data = ((signal_t1.data * signal_t1.data_std) + signal_t1.data_mean) - ((signal_t0.data * signal_t0.data_std) + signal_t0.data_mean) 
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data = (self.data - self.data_mean) / self.data_std
        logging.info(f"Compiled signald data for shot {shotnr}, mean={self.data_mean}, std={self.data_std}")




class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, shotnr, tstart, tend, tsample, shift_target=10, 
                 predictors=["pinj", "neutrons", "q95_prof"],
                 targets=["ae_prob_delta"],
                 datapath="/projects/EKOLEMEN/aza_lenny_data1",
                 device="cpu"):
        """Initializes the dataloader for DIII-D data.

        Input:
        =======
        super().__inite__(self
        shotnr......: (int) shot number
        tstart......: (float) Start time, in milliseconds
        tend........: (float) End time, in milliseconds
        tsample.....: (float) Time between samples, in milliseconds
        shift_target: (float) Defines shift in time for the targets, in milliseconds
        device......: (string) device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # Directory where ECE, Profiles, and Pnbi data are stored
        self.datapath = datapath
        # Shot number
        self.shotnr = shotnr
        # Set up sampling interval for data.
        self.tstart = tstart
        self.tend = tend
        self.tsample = tsample # Sub-sample all signals at this frequency
        self.shift_target = shift_target

        logging.info(f"Using device {device}")

        assert(tstart < tend)
        assert((tend - tstart) > tsample)
        self.predictors = {}
        self.targets = {}

        # Initialize all predictors
        for pred_name in predictors:
            if pred_name == "pinj":
                logging.info("Adding pinj to predictor list")
                self.predictors["pinj"] = signal_pinj(shotnr, tstart, tend, tsample, 
                                                      override_dt=0.01, device=device)
            if pred_name == "ae_prob":
                logging.info("Adding ae_prob to predictor list")
                self.predictors["ae_prob"] = signal_ae_prob(shotnr, tstart, tend, tsample, 
                                                            override_dt=0.002, device=device)
            if pred_name == "neut":
                logging.info("Adding neutron rate to predictor list")
                self.predictors["neut"] = signal_neut(shotnr, tstart, tend, tsample, 
                                                      override_dt=2e-2, device=device)


        for target_name in targets:
            if target_name == "ae_prob_delta":
                self.targets["ae_prob_delta"] = signal_ae_prob_delta(shotnr, tstart, tend, tsample,
                                                                     tshift=shift_target, 
                                                                     device=device)

        # Assert that all data has the same number of samples
        base_key = next(iter(self.predictors.keys()))
        for k in self.predictors.keys():
            assert(self.predictors[k].data.shape[0] == self.predictors[base_key].data.shape[0])

        for k in self.targets.keys():
            assert(self.targets[k].data.shape[0] == self.predictors[base_key].data.shape[0])


    def __len__(self):
        # Returns the number of time samples
        k = next(iter(self.predictors.keys()))
        return self.predictors[k].data.shape[0]

   
    def __getitem__(self, idx):
        """Fetch data corresponding to the idx'th sample."""
        data_t0 = torch.cat((self.predictors['ae_prob'][idx, :], 
                             self.predictors['neut'][idx], 
                             self.predictors['pinj'][idx]))
        data_t1 = self.targets['ae_prob_delta'][idx, :]
        return (data_t0, data_t1)
#
# End of file d3d_loaders.py
