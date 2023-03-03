#-*- encoding: utf-8 -*-

"""Contains class definitions used to as abstractions for 1d signals."""

from os.path import join
import time

import numpy as np
import h5py
import torch

import logging

class signal_1d():
    """Base class for a 1d sample.

    Represents a 1-d signal over [tstart:tend].
    Aims to use data stored in /projects/EKOLEMEN/aza_lenny_data1/template.
    Check README for currently supported signals.


    """
    def __init__(self, shotnr, t_params, 
            datapath="/projects/EKOLEMEN/d3d_loader",
            device="cpu"):
        """Load data from HDF5 file, standardize, and move to device.

        Parameters
        ----------
        shotnr : int
                 Shot number

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
        self.data = (self.data - self.data_mean) / (self.data_std + 1e-10)
        logging.info(f"""Compiled signal {self.__class__.__name__}, 
                         tstart={self.tstart}, tend={self.tend}, tsample={self.tsample}, tshift={self.tshift},
                         datapath={self.datapath}, 
                         mean={self.data_mean}, std={self.data_std}""")

    def _get_time_sampling(self, tb):
        """
        Use the time base to calculate the closest indices to desired tsample
        
        NOTE: 
        If tsample is set to -1, will return the full signal between tstart and tend
        
        Parameters
        ----------
        tb : float array
                time base array, the time each sample corresponds to.
        
        Returns
        -------
        t_inds: int array
                    Index of closest measurement to desired time
                    (desired time is a ceiling so no data from future)
        """      
        # Return full signal if tsample is -1
        if self.tsample == -1:
            # Finds closest start and end indices, forward or backwards in time
            tstart_ind = np.searchsorted(tb, self.tstart)
            tend_ind = np.searchsorted(tb, self.tend)
            return np.arange(tstart_ind, tend_ind)
         
        # Forced sampling times
        time_samp_vals = np.arange(self.tstart, self.tend, self.tsample)
        
        # Shift times. To have samples at a future time (shifted by tshift) line up
        # at the same index as the unshifted time, we need to subtract the time shift
        # from the time-base.
        

        time_samp_vals -= self.tshift
        if self.tshift > 0.0:
            logging.info(f"shifted time_samp_vals = {time_samp_vals}")

        tb_ind = 1 # Index of time in tb
        num_samples = len(time_samp_vals)
        t_inds = np.zeros((num_samples,), dtype=int)
        
        # Raise error if first time sample comes before first measurement
        if tb[0] > time_samp_vals[0]:
            raise(ValueError(f'Time of first requested sample is before first real measurement was taken for {self.name}'))
        
        for i, time_samp in enumerate(time_samp_vals):
            # Scan the signals time_base as long as we are below the next desired sampling time
            while tb[tb_ind] < time_samp and tb_ind < len(tb) - 1:
                tb_ind += 1
            
            # Save last index where tb[tb_ind] < time_samp
            t_inds[i] = tb_ind - 1

        return t_inds

    def _cache_data(self):
        """Default reader to cache data from hdf5 file. 
        
        Works for all the 1d signals except for pinj. 
        This function is overwritten by pinj and 2d signals

        Returns
        -------
        data : tensor
                        Data time series of diagnostic. dim0: samples. dim1: features

        Raises
        ------
        ValueError
            When the timebase can't be loaded from HDF we assume that there is no data as well.
        """
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        # Don't use with... scope. This throws off dataloader
        fp = h5py.File(join(self.datapath, f"{self.shotnr}.h5")) 
        # Checks to make sure predictor is present
        try:
            tb = torch.tensor(fp[self.key]["xdata"][:])
        except ValueError as e:
            logging.error(f"Unable to load timebase for shot {self.shotnr} signal {self.name}")
            raise e

        # Some shots have no data for a given signal. In that case, the tensor is present in the
        # dataset but the size is 0. Throw an error if that is the case.
        if tb.shape[0] < 2:
            raise ValueError(f"Shot {self.shotnr}, signal {self.key}: Timebase in HDF5 file has length {tb.shape[0]} < 2!")
        
        # Indices to sample on
        t_inds = self._get_time_sampling(tb)        
        data = torch.tensor(fp[self.key]["zdata"][:])[t_inds]
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
    """Sum of total injected power.
    
        Args:
        -----
            name (str):
            collect_keys (list(str))): List of datasets in the HDF5 file that will be summed to build the output signal.
        

        This signal is constructed by summing over a list of neutral beam injected power time
        series. The list of signal over which we sum is given by the collect_keys argument.
    """
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.name = 'pinj'
        self.collect_keys = ["pinjf_15l", "pinjf_15r", "pinjf_21l", "pinjf_21r", 
                             "pinjf_30l", "pinjf_30r", "pinjf_33l", "pinjf_33r"]
        super().__init__(shotnr, t_params, datapath, device)
    
    def _cache_data(self):
        """Load sum of all pinj from hdf5 data file.
        
        Returns
        -------
        pinj_data : tensor
                    Data time series of sum over all Pinj nodes. dim0: samples. dim1: features
        """
        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        fp = h5py.File(join(self.datapath, f"{self.shotnr}.h5")) 
        # Collect the time base using the 15l signal
        tb = torch.tensor(fp["pinjf_15l"]["xdata"][:]) # Get time-base
        t_inds = self._get_time_sampling(tb)
        # Sum the contributions from all neutral beams specified in the collect_keys list.
        pinj_data = sum([torch.tensor(fp[k]["zdata"][:])[t_inds] for k in self.collect_keys])
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Caching pinj data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
           
        return pinj_data.unsqueeze(1)


class signal_tinj(signal_1d):
    """Injected torque"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "tinj"
        self.name = "tinj"
        super().__init__(shotnr, t_params, datapath, device)


class signal_bmspinj(signal_1d):
    """Injected torque"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "bmspinj"
        self.name = "bmspinj"
        super().__init__(shotnr, t_params, datapath, device)


class signal_bmstinj(signal_1d):
    """Injected torque"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "bmstinj"
        self.name = "bmstinj"
        super().__init__(shotnr, t_params, datapath, device)


class signal_neut(signal_1d):
    """Neutrons rate 1d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "neutronsrate"
        self.name = "neutronsrate"
        super().__init__(shotnr, t_params, datapath, device)


class signal_iptipp(signal_1d):
    """Target current"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "iptipp"
        self.name = "iptipp"
        super().__init__(shotnr, t_params, datapath, device)


class signal_dstdenp(signal_1d):
    """Target density"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "dstdenp"
        self.name = "dstdenp"
        super().__init__(shotnr, t_params, datapath, device)


class signal_dssdenest(signal_1d):
    """Line-averaged density"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "dssdenest"
        self.name = "dssdenest"
        super().__init__(shotnr, t_params, datapath, device)


class signal_ip(signal_1d):
    """Injected power 1d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "ip"
        self.name = "ip"
        super().__init__(shotnr, t_params, datapath, device)


class signal_betan(signal_1d):
    """Injected power 1d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "betan"
        self.name = "betan"
        super().__init__(shotnr, t_params, datapath, device)



class signal_doutl(signal_1d):
    """Lower triangularity shape profile"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "doutl"
        self.name = "lower triangularity"
        super().__init__(shotnr, t_params, datapath, device)


class signal_doutu(signal_1d):
    """Upper triangularity shape profile"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "doutu"
        self.name = "upper triangularity"
        super().__init__(shotnr, t_params, datapath, device)

class signal_tritop(signal_1d):
    """Lower triangularity shape profile"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "tritop"
        self.name = "lower triangularity"
        super().__init__(shotnr, t_params, datapath, device)


class signal_tribot(signal_1d):
    """Upper triangularity shape profile"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "tribot"
        self.name = "lower triangularity"
        super().__init__(shotnr, t_params, datapath, device)


class signal_ech(signal_1d):
    """ECH 1d signal. Uses the corrected echpwrc pointname"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "echpwrc"
        self.name = "ECH"
        super().__init__(shotnr, t_params, datapath, device)


class signal_q95(signal_1d):
    """q95 value - 1d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "q95"
        self.name = "q95"
        super().__init__(shotnr, t_params, datapath, device)
    
    
class signal_kappa(signal_1d):
    """Shape 1d signal Kappa"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3d_loader", device="cpu"):
        self.key = "kappa"
        self.name = "kappa"
        super().__init__(shotnr, t_params, datapath, device)

class signal_pcbcoil(signal_1d):
    """PCBcoil signal."""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMENT/d3d_loader", device="cpu"):
        self.key = "pcbcoil"
        self.name = "pcbcoil"
        super().__init__(shotnr, t_params, datapath, device)
