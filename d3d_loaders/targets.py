#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join
import h5py
import numpy as np
import torch 
import logging


""" TODO: Make a class factory for targets, just like signal_factory
in signal0d.py

"""

class target_ttd():
    """Base class for targets.

    Hard-coded TTD target. Should be extended for TTELM

    Things like time-to-disruption, time-to-ELM, etc.
    """
    def __init__(self, shotnr, time_sampler, datapath, device=torch.device("cpu")):
        """Load target from HDF5 file, resample move to device.

        Parameters:
            shotnr : int
            time_sampler : class `causal_sampler`
            datapath : string
            device : torch.device("cuda:0" if torch.cuda_is_available() else "cpu")
        """
        self.shotnr = shotnr
        self.time_sampler = time_sampler
        self.datapath = datapath

        tb, target = self._cache_data()
        data = self.time_sampler.resample(tb, target)
        # Convert back to torch, since the sampler uses numpy/scipy arrays.
        # Un-squeeze observation dimension
        self.data = torch.tensor(data).unsqueeze(1).to(device)

    def _cache_data(self):
        """Fetchd data from HDF5. """
        fp = h5py.File(join(self.datapath, f"{self.shotnr}.h5"), "r")
        try:
            tb = torch.tensor(fp["target_ttd"]["xdata"][:])
        except ValueError as e:
            fp.close()
            logging.error(f"Unable to load timebase for shot {self.shotnr} for target_ttd")
            raise e
        
        if tb.shape[0] < 2:
            fp.close()
            raise ValueError(f"Shot {self.shotnr}, target_ttd: Timebase in HDF5 file has length {tb.shape[0]} < 2!")    
        
        target = torch.tensor(fp["target_ttd"]["zdata"][:])
        fp.close()

        return tb, target
    

class target_ttelm():
    """Time-to-ELM target

    Time-to-ELM target
    """
    def __init__(self, shotnr, time_sampler, datapath, device=torch.device("cpu")):
        """Load target from HDF5 file, resample move to device.

        Parameters:
            shotnr : int
            time_sampler : class `causal_sampler`
            datapath : string
            device : torch.device("cuda:0" if torch.cuda_is_available() else "cpu")
        """
        self.shotnr = shotnr
        self.time_sampler = time_sampler
        self.datapath = datapath

        tb, target = self._cache_data()
        data = self.time_sampler.resample(tb, target)
        # Convert back to torch, since the sampler uses numpy/scipy arrays.
        # Un-squeeze observation dimension
        self.data = torch.tensor(data).unsqueeze(1).to(device)

    def _cache_data(self):
        """Fetchd data from HDF5. """
        fp = h5py.File(join(self.datapath, f"{self.shotnr}.h5"), "r")
        try:
            tb = torch.tensor(fp["target_ttelm"]["xdata"][:])
        except ValueError as e:
            fp.close()
            logging.error(f"Unable to load timebase for shot {self.shotnr} for target_ttelm")
            raise e
        
        if tb.shape[0] < 2:
            fp.close()
            raise ValueError(f"Shot {self.shotnr}, target_ttd: Timebase in HDF5 file has length {tb.shape[0]} < 2!")    
        
        target = torch.tensor(fp["target_ttelm"]["zdata"][:])
        fp.close()

        return tb, target


# end of file targets.py