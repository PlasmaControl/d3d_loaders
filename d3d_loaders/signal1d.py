#-*- encoding: utf-8 -*-

"""Contains class definitions used to as abstractions for 1d signals."""

from os.path import join
from math import ceil
import time

import numpy as np
import h5py
import torch

import logging


class signal_1d():
    """Base class for a 1d sample.

    Represents a 1-d signal over [tstart:tend].
    Aims to use data stored in /projects/EKOLEMEN/aza_lenny_data1/template.
    Currently supports only ece, pinj, and other data contained in profiles.h5


    """
    def __init__(self, shotnr, t_params, 
            datapath="/projects/EKOLEMEN/aza_lenny_data1",
            device="cpu"):
        """Load data from HDF5 file, standardize, and move to device.

        Parameters
        ----------
        shotnr : Array of ints or single int
                 Shot numbers

        t_params : dict
                   Contains the following necessary keys:
                 
                        tstart : float
                                    Start of signal interval, in milliseconds
                        tend : float
                                End of signal interval, in milliseconds
                        tsample : float
                                    Desired sampling time, in milliseconds
                                    
                   Optional keys/arguments:
                        tshift : float, default=0.0
                                Shift signal by tshift with respect to tstart, in milliseconds
                                    
        datapath : string, default='/projects/EKOLEMEN/aza_lenny_data1'
                   Basepath where HDF5 data is stored.  
        device : string, default='cpu'
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        # Turn shotnr to array if single int
        if type(shotnr) == int:
            shotnr = [shotnr]
            
        # Store function arguments as member variables
        self.shotnr = shotnr
        self.tstart = t_params["tstart"]
        self.tend = t_params["tend"]
        self.tsample = t_params["tsample"]
        try:
            self.tshift = t_params["tshift"]
        except:
            self.tshift = 0.0
        self.datapath = datapath
        
        # Load data from HDF5 file and store, move to device
        self.data = self._cache_data().to(device)
        # Z-score normalization
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data = (self.data - self.data_mean) / self.data_std
        logging.info(f"""Compiled signal {self.__class__.__name__}, 
                         tstart={self.tstart}, tend={self.tend}, tsample={self.tsample}, tshift={self.tshift},
                         datapath={self.datapath}, 
                         mean={self.data_mean}, std={self.data_std}""")

    def _get_time_sampling(self, tb):
        """
        Use the time base to calculate the closest indices to desired tsample
        
        Parameters
        ----------
        tb : float array
                time base array, the true time of each measurement
        
        Returns
        -------
        t_inds: int array
                    Index of closest measurement to desired time
                    (desired time is a ceiling so no data from future)
        """       
        # Forced sampling times
        time_samp_vals = np.arange(self.tstart, self.tend, self.tsample)
        
        # Shift times
        time_samp_vals += self.tshift
        
        tb_ind = 1 # Index of time in tb
        num_samples = len(time_samp_vals)
        t_inds = np.zeros((num_samples,), dtype=int)
        
        # Make sure our first sample is after first true measurement was taken
        assert tb[0] < time_samp_vals[0]
        
        for i, time_samp in enumerate(time_samp_vals):
            # Increase tb_ind until the real time is past our sampling time
            while tb[tb_ind] < time_samp and tb_ind < len(tb) - 1:
                tb_ind += 1
            
            # Save last index where tb[tb_ind] < time_samp
            t_inds[i] = tb_ind - 1

        return t_inds

    def _cache_data(self):
        """Default reader to cache data from hdf5 file. 
        
        Works for neutrons, ip, ech, and kappa. This function is overwritten by pinj and 2d signals

        Returns
        -------
        data : tensor
                        Data time series of diagnostic. dim0: samples. dim1: features
        """
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        # Don't use with... scope. This throws off dataloader
        data = None
        for shot in self.shotnr:
            fp = h5py.File(join(self.datapath, "template", f"{shot}_{self.file_label}.h5")) 
            tb = torch.tensor(fp[self.key]["xdata"][:])

            t_inds = self._get_time_sampling(tb)
            
            if data == None:
                data = torch.tensor(fp[self.key]["zdata"][:])[t_inds]
            else:
                # Append new shot along samples axis (axis=0)
                data = torch.cat((data, torch.tensor(fp[self.key]["zdata"][:])[t_inds]), 0)
            fp.close()

        elapsed = time.time() - t0_p       
        logging.info(f"Caching {self.name} data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")

        return data.unsqueeze(1)

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
                    Data time series of sum over all Pinj nodes. dim0: samples. dim1: features
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        pinj_data = None
        for shot in self.shotnr:
            fp = h5py.File(join(self.datapath, "template", f"{shot}_pinj.h5")) 
            tb = torch.tensor(fp["pinjf_15l"]["xdata"][:]) # Get time-base

            t_inds = self._get_time_sampling(tb)
            if pinj_data == None:
                pinj_data = sum([torch.tensor(fp[k]["zdata"][:])[t_inds] for k in fp.keys()])
            else:
                pinj_data = torch.cat((pinj_data, sum([torch.tensor(fp[k]["zdata"][:])[t_inds] for k in fp.keys()])), 0)
            fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Caching pinj data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
           
        return pinj_data.unsqueeze(1)


class signal_neut(signal_1d):
    "Neutrons rate 1d signal"
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        self.key = "neutronsrate"
        self.file_label = "profiles"
        self.name = "neutron"
        super().__init__(shotnr, t_params, datapath, device)


class signal_ip(signal_1d):
    "ip 1d signal"
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        self.key = "ip"
        self.file_label = "profiles"
        self.name = "ip"
        super().__init__(shotnr, t_params, datapath, device)
   
    
class signal_ech(signal_1d):
    "ECH 1d signal. Uses the corrected echpwrc pointname"
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        self.key = "echpwrc"
        self.file_label = "ech"
        self.name = "ECH"
        super().__init__(shotnr, t_params, datapath, device)
    
class signal_q95(signal_1d):
    """q95 value - 1d signal
    
    """
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        self.key = "q95"
        self.file_label = "profiles"
        self.name = "q95"
        super().__init__(shotnr, t_params, datapath, device)
    
    
class signal_kappa(signal_1d):
    "Shape 1d signal Kappa"
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/aza_lenny_data1", device="cpu"):
        self.key = "kappa"
        self.file_label = "shape"
        self.name = "kappa"
        super().__init__(shotnr, t_params, datapath, device)