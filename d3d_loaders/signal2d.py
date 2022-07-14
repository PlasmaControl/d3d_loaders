#-*- encoding: utf-8 -*-

"""Contains class definitions used to as abstractions for 2d signals."""

from os.path import join
from math import ceil
import time
import pickle
import numpy as np

import h5py
import torch

import logging
from signal1d import signal_1d

from d3d_loaders.rcn_functions import rcn_infer


class signal_2d(signal_1d):
    """Base class for a 2d sample. Subclass of 1d signal class. 

    Represents a 2-d signal over [tstart:tend].
    Aims to use data stored in /projects/EKOLEMEN/aza_lenny_data1/template.
    Currently supports only ece, pinj, and other data contained in profiles.h5

    """    

    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: profile length. dim1: samples
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_{self.file_label}.h5")) 
        tb = torch.tensor(fp[self.key]["xdata"][:]) # Get time-base
        
        t0_idx, shift_smp, num_samples, nth_sample = self._get_time_sampling(tb)
        
        logging.info(f"Sampling {self.name}: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        prof_data = torch.tensor(fp[self.key]["zdata"][:])[t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample,:]
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading {self.name} for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        return prof_data


class signal_dens(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "edensfit"
        self.file_label = "profiles"
        self.name = "dens"


class signal_temp(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "etempfit"
        self.file_label = "profiles"
        self.name = "temp"


class signal_pres(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "pres"
        self.file_label = "profiles"
        self.name = "pres"
    
    
class signal_q(signal_2d):
    """q profile - 2d signal
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "q"
        self.file_label = "profiles"
        self.name = "q"
    

class signal_q95(signal_2d):
    """q95 profile - 2d signal
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "q95"
        self.file_label = "profiles"
        self.name = "q95"


class signal_ae_prob(signal_2d):
    """Probability for a given Alfven Eigenmode."""
    def __init__(self, shotnr, tstart, tend, tsample, 
            tshift=0.0, override_dt=None, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Loads weights for RCN model and calls base class constructor.
        
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

        if self.override_dt is None:
            self.dt = np.diff(tb).mean()
        else:
            self.dt = self.override_dt        
        num_samples, nth_sample = self._get_num_n_samples(self.dt)
        
        shift_smp = int(ceil(self.tshift/ self.dt))
        t0_idx = np.argmin(np.abs(tb - self.tstart))

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

        logging.info(f"Caching AE forward model using RCN, {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")

        return torch.tensor(ae_probs)


class signal_ae_prob_delta(signal_2d):
    """Change in Alfven Eigenmode probability over time""" 
    def __init__(self, shotnr, tstart, tend, tsample, 
            tshift=0.0, override_dt=None, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Construct difference in AE probability using two signal_ae_prob.
        
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

        Constructs a the change in Alfven Eigenmode probability using a finite difference method.
        The signals are separated by t_shift.
        """
        
        
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


class signal_tri_l(signal_2d):
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "triangularity_l"
        self.file_label = "shape"
        self.name = "lower triangularity"
        

class signal_tri_u(signal_2d):
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "triangularity_u"
        self.file_label = "shape"
        self.name = "upper triangularity"


class signal_ece(signal_2d):
    """Raw ECE signals

    Returns
    -------
    ece_data : tensor
                Data time series for profiles. dim0: ECE channels. dim1: samples
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, 
                 datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu",
                 channels=range(1,41)):
        """
        Unique part of constructor is channels. Can be any list of numbers from 1-40, or 
        just an individual channel. 
        """
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.channels = channels
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: profile length. dim1: samples
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_ece.h5")) 
        tb = torch.tensor(fp['ece']["xdata"][:]) # Get time-base
        
        t0_idx, shift_smp, num_samples, nth_sample = self._get_time_sampling(tb)
        
        logging.info(f"Sampling {self.name}: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        # Load and stack ECE channels, slicing happens in for loop to avoid loading data that would then be cut
        prof_data = torch.tensor(np.stack([fp['ece'][f"tecef{channel:02d}"]
                                           [t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample,:] for channel in self.channels]))
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading {self.name} for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        # NOTE: unsqueeze(1) not needed even if there's only 1 channel 
        return prof_data


class signal_ece_spec(signal_2d):
    """_summary_
    
    """
    def _cache_data(self):
        # Fill in later
        pass


class signal_co2_spec(signal_2d):
    """_summary_

    """
    def _cache_data(self):
        # Fill in later
        pass

