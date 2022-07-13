#-*- encoding: utf-8 -*-

"""Contains class definitions used to as abstractions for 1d signals."""

from os.path import join
from math import ceil
import time
import pickle

import numpy as np
import h5py
import torch

import logging

from d3d_loaders.rcn_functions import rcn_infer


class signal_1d():
    """Base class for a 1d sample.

    Represents a 1-d signal over [tstart:tend].
    Aims to use data stored in /projects/EKOLEMEN/aza_lenny_data1/template.
    Currently supports only ece, pinj, and other data contained in profiles.h5


    """
    def __init__(self, shotnr, tstart, tend, tsample, 
            tshift=0.0, override_dt=None, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Load data from HDF5 file, standardize, and move to device.

        Parameters
        ----------
        shotnr : Int
                 Shot number
        tstart : float
                 Start of signal interval, in milliseconds
        tend : float
               End of signal interval, in milliseconds
        tsample : float
                  Desired sampling time, in milliseconds
        tshift : float, default=0.0
                 Shift signal by tshift with respect to tstart, in milliseconds
        override_dt : float, optional
                      Use this value as sample spacing instead of calculating from xdata field in HDF5 file
        datapath : string, default='/projects/EKOLEMEN/aza_lenny_data1'
                   Basepath where HDF5 data is stored.  
        device : string, default='cpu'
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        # Store function arguments as member variables
        self.shotnr = shotnr
        self.tstart = tstart
        self.tend = tend
        self.tsample = tsample
        self.tshift = tshift
        self.override_dt = override_dt
        self.datapath = datapath
        
        # Load data from HDF5 file and store, move to device
        self.data = self._cache_data().to(device)
        # Z-score normalization
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data = (self.data - self.data_mean) / self.data_std
        logging.info(f"Compiled signal data for shot {shotnr}, mean={self.data_mean}, std={self.data_std}")


    def _get_num_n_samples(self, dt):
        """Calculates number of samples and sample skipping.

        This function is 

        Given a signal sampled on [tstart:tend] with sample spacing dt
        calculate the total number of samples available.

        Also calculates the sample skipping when sub-sampling dt to self.tsample.

        Parameters
        ----------
        dt : sampling time, in milliseconds

        Returns
        -------
        num_samples : int
                      Number of samples that cover the interval [tstart:tend]
        nth_sample : int
                     Number of samples that are skipped in the original time series 
        """

        num_samples = int(ceil((self.tend - self.tstart) / dt))
        nth_sample = int(ceil(self.tsample / dt))

        return num_samples, nth_sample

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
        """Load sum of all pinj from hdf5 data file.
        
        Returns
        -------
        pinj_data : tensor
                    Data time series of sum over all Pinj nodes. dim0: features. dim1: samples
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_pinj.h5")) 
        tb = torch.tensor(fp["pinjf_15l"]["xdata"][:]) # Get time-base

        if self.override_dt is None:
            self.dt = (tb[1:] - tb[:-1]).mean()           # Get sampling time
        else:
            self.dt = self.override_dt
        # Get total number of samples and desired sub-sample spacing
        num_samples, nth_sample = self._get_num_n_samples(self.dt)
        shift_smp = int(ceil(self.tshift/ self.dt))
        t0_idx = torch.argmin(torch.abs(tb - self.tstart))
        logging.info(f"Sampling pinj: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        pinj_data = sum([torch.tensor(fp[k]["zdata"][:])[t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample] for k in fp.keys()])
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading pinj for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
           
        return pinj_data.unsqueeze(1)


class signal_neut(signal_1d):
    def _cache_data(self):
        """Load neutron emission rate from hdf5 data file
        
        Returns
        -------
        neutron_data : tensor
                       Data time series of neutron rate. dim0: features. dim1: samples
        """
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        # Don't use with... scope. This throws off dataloader
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_profiles.h5")) 
    
        tb = torch.tensor(fp["neutronsrate"]["xdata"][:])
        if self.override_dt is None:
            self.dt = np.diff(tb).mean()
        else:
            self.dt = self.override_dt

        num_samples, nth_sample = self._get_num_n_samples(self.dt)  # Number of samples and sample spacing
        shift_smp = int(ceil(self.tshift/ self.dt))                  # Shift samples by this number into the fuiture
        t0_idx = torch.argmin(torch.abs(tb - self.tstart))
        logging.info(f"Sampling neut: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        neutron_data = torch.tensor(fp["neutronsrate"]["zdata"][t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample])
        fp.close()    

        elapsed = time.time() - t0_p       
        logging.info(f"Loading neutron data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")


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

        The RCN model is used to translate ECE data to Alfven Eigenmod probability.
        This has to be done sequentially, i.e. by iterating over all samples.
        
        
        Returns
        -------
        ae_probs : tensor
                   Probability for presence of an Alfven Eigenmode. dim0: feature. dim1: sample
        """

        # Find how many samples apart tsample is
        t0_p = time.time()
        # Don't use scope. This throws off multi-threaded loaders
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_ece.h5"), "r") 
        tb = fp["ece"]["xdata"][:]    # Get ECE time-base
        dt = np.diff(tb).mean()   # Get ECE sampling time
        num_samples, nth_sample = self._get_num_n_samples(dt)
        
        shift_smp = int(ceil(self.tshift/ self.dt))
        t0_idx = np.argmin(np.abs(tb - self.tstart))
        logging.info(f"Sampling ECE: t0_idx={t0_idx}, dt={dt}, num_samples={num_samples}, nth_sample={nth_sample}, shift_smp={shift_smp}")

        # Read in all ece_data at t0 and shifted at t0 + 50 mus
        ece_data = np.vstack([fp["ece"][f"tecef{(i+1):02d}"][t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample ] for i in range(40)]).T
        fp.close()
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
        logging.info(f"Compiled signal data for shot {shotnr}, mean={self.data_mean}, std={self.data_std}")



