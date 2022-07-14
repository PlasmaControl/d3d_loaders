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
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_profile.h5")) 
        tb = torch.tensor(fp[self.key]["xdata"][:]) # Get time-base
        
        t0_idx, shift_smp, num_samples, nth_sample = self._get_time_sampling(tb)
        
        logging.info(f"Sampling {self.name}: t0_idx={t0_idx}, dt={self.dt}, num_samples={num_samples}, nth_sample={nth_sample}")

        prof_data = torch.tensor(fp[self.key]["zdata"][:])[t0_idx + shift_smp:t0_idx + shift_smp + num_samples:nth_sample]
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading {self.name} for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
           
        return prof_data.unsqueeze(1)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"signal_1d({self.data}"
    
    def _get_time_sampling(self, tb):
        """
        Use the time base to calculate the variables for indexing 2d signal
        
        NOTE: This will be changed or moved since time sampling needs to be refined
        """
        if self.override_dt is None:
            self.dt = (tb[1:] - tb[:-1]).mean()           # Get sampling time
        else:
            self.dt = self.override_dt
        # Get total number of samples and desired sub-sample spacing
        num_samples, nth_sample = self._get_num_n_samples(self.dt)
        shift_smp = int(ceil(self.tshift/ self.dt))
        t0_idx = torch.argmin(torch.abs(tb - self.tstart))
        
        return t0_idx, shift_smp, num_samples, nth_sample


class signal_dens(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "edensfit"
        self.name = "dens"
    
class signal_temp(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "etempfit"
        self.name = "temp"


class signal_pres(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "pres"
        self.name = "pres"
    
    
class signal_q(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "q"
        self.name = "q"
    

class signal_q95(signal_2d):
    """_summary_
    
    """
    def __init__(self, shotnr, tstart, tend, tsample, tshift=0, override_dt=None, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        super().__init__(shotnr, tstart, tend, tsample, tshift, override_dt, datapath, device)
        self.key = "q95"
        self.name = "q95"


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

