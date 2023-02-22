#-*- encoding: utf-8 -*-

"""Contains class definitions used to as abstractions for 2d signals."""

from os.path import join
import time
import pickle
import numpy as np

import h5py
import torch

import logging
import sys
sys.path.append("..")

from d3d_loaders.signal1d import signal_1d
from d3d_loaders.rcn_functions import rcn_infer


class signal_2d(signal_1d):
    """Base class for a 2d sample. Subclass of 1d signal class. 

    Represents a 2-d signal over [tstart:tend].
    Aims to use data stored in /projects/EKOLEMEN/d3dloader.
    Check README for currently supported signals.

    """    

    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples dim1: profile length
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        fp = h5py.File(join(self.datapath, f"{shotnr}.h5")) 
        try:
            tb = fp[self.key]["xdata"][:] # Get time-base
        except ValueError as e:
            logging.error(f"Unable to load timebase for shot {self.shotnr} signal {self.name}")
            raise e
        
        t_inds = self._get_time_sampling(tb)
        prof_data = (torch.tensor(fp[self.key]["zdata"][:]).T)[t_inds,:]
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading {self.name}, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        return prof_data


class signal_dens(signal_2d):
    """Density profile - 2d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3dloader", device="cpu"):
        self.key = "edensfit"
        self.file_label = "profiles"
        self.name = "dens"
        super().__init__(shotnr, t_params, datapath, device)


class signal_temp(signal_2d):
    """Electron Temperature profile - 2d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3dloader", device="cpu"):
        self.key = "etempfit"
        self.file_label = "profiles"
        self.name = "temp"
        super().__init__(shotnr, t_params, datapath, device)


class signal_pres(signal_2d):
    """Pressure profile - 2d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3dloader", device="cpu"):
        self.key = "pres"
        self.file_label = "profiles"
        self.name = "pres"
        super().__init__(shotnr, t_params, datapath, device)
    
    
class signal_q(signal_2d):
    """q profile - 2d signal"""
    def __init__(self, shotnr, t_params, datapath="/projects/EKOLEMEN/d3dloader", device="cpu"):
        self.key = "q"
        self.file_label = "profiles"
        self.name = "q"
        super().__init__(shotnr, t_params, datapath, device)


class signal_ae_prob(signal_2d):
    """Probability for a given Alfven Eigenmode."""
    def __init__(self, shotnr, t_params, 
            datapath="/projects/EKOLEMEN/d3dloader",
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
        datapath : string, default='/projects/EKOLEMEN/aza_lenny_data1'
                   Basepath where HDF5 data is stored.  
        device : string, default='cpu'
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        self.name = 'ae_prob'
        # The RCN model weights are needed to call the base-class constructor
        with open('/home/rkube/ml4control/1dsignal-model-AE-ECE-RCN.pkl', 'rb') as df:
            self.infer_data = pickle.load(df)
        self.n_res_l1 = self.infer_data['layer1']['w_in'].shape[0]
        self.n_res_l2 = self.infer_data['layer2']['w_in'].shape[0]

        # Call base class constructor to fetch and store data
        super().__init__(shotnr, t_params, datapath, device)


    def _cache_data(self):
        """Loads ECE data and calculates AE probability using Aza's RCN model.

        The RCN model is used to translate ECE data to Alfven Eigenmod probability.
        This has to be done sequentially, i.e. by iterating over all samples.
        
        
        Returns
        -------
        ae_probs : tensor
                   Probability for presence of an Alfven Eigenmode. dim0: samples. dim1: features
        """
        # Find how many samples apart tsample is
        t0_p = time.time()
        # Don't use scope. This throws off multi-threaded loaders
        fname = join(self.datapath, f"{self.shotnr}.h5")
        fp = h5py.File(fname, "r") 
        tb = fp["tecef01"]["xdata"][:]    # Get ECE time-base
        t_inds = self._get_time_sampling(tb)
        # Read in ece_data  as numpy array for consumption 
        ece_data = np.vstack([fp[f"tecef{i:02d}"]["zdata"][t_inds] for i in range(1, 41)]).T
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
        #print(ece_data.shape)
        # Iterate over time index 0
        for idx, u in enumerate(ece_data):
            L = "layer1"
            #print(idx, u.shape)
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
    def __init__(self, shotnr, t_params, 
            datapath="/projects/EKOLEMEN/d3dloader",
            device="cpu"):
        """Construct difference in AE probability using two signal_ae_prob.
        
        Parameters
        ----------
        shotnr : Int
                 Shot number
        t_params: dict
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
        datapath : string, default='/projects/EKOLEMEN/d3dloader'
                   Basepath where HDF5 data is stored.  
        device : string, default='cpu'
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        Constructs a the change in Alfven Eigenmode probability using a finite difference method.
        The signals are separated by t_shift.
        """
        self.name = 'ae_prob_delta'
        



        self.shotnr = shotnr
        self.tstart = t_params["tstart"]
        self.tend = t_params["tend"]
        self.tsample = t_params["tsample"]
        if("tshift" not in t_params.keys()):
            raise(KeyError("dictionary key 'tshift' not found when constructing `signal_ae_prob_delta`"))

        self.datapath = datapath
        self.data = ((signal_t1.data * signal_t1.data_std) + signal_t1.data_mean) - ((signal_t0.data * signal_t0.data_std) + signal_t0.data_mean) 
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data = (self.data - self.data_mean) / (self.data_std + 1e-10)

        # Shifted signal
        signal_t1 = signal_ae_prob(shotnr, t_params, datapath=datapath, device=device)
        # Signal at t0, set t_shift = 0
        t_params["tshift"] = 0.0
        signal_t0 = signal_ae_prob(shotnr, t_params, datapath=datapath, device=device)

        logging.info(f"Compiled signal data for shot {shotnr}, mean={self.data_mean}, std={self.data_std}")


class signal_ece(signal_2d):
    """Raw ECE signals

    Returns
    -------
    ece_data : tensor
                Data time series for profiles. dim0: ECE channels. dim1: samples
    """
    def __init__(self, shotnr, t_params, 
                 datapath="/projects/EKOLEMEN/d3dloader", device="cpu",
                 channels=range(1,41)):
        """
        Unique part of constructor is channels. Can be any list of numbers from 1-40, or 
        just an individual channel. 
        """
        self.channels = channels
        self.name = 'raw ece'
        super().__init__(shotnr, t_params, datapath, device)
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples. dim1: features (channels)
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        fp = h5py.File(join(self.datapath, f"{self.shotnr}.h5")) 
        tb = torch.tensor(fp['tecef01']["xdata"][:]) # Get time-base
        
        t_inds = self._get_time_sampling(tb)

        # Load and stack ECE channels, slicing happens in for loop to avoid loading data that would then be cut
        prof_data = torch.tensor(np.stack([fp[f"tecef{channel:02d}"]["zdata"][t_inds] for channel in self.channels], axis=1))
        fp.close()
        elapsed = time.time() - t0_p
        logging.info(f"Loading raw ECE, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        # NOTE: unsqueeze(1) not needed even if there's only 1 channel 
        return prof_data


class signal_co2_dp(signal_2d):
    """Raw CO2 signals (change in CO2 phase data)

    Returns
    -------
    co2_dp_data : tensor
                Data time series for profiles. dim0: co2 signals. dim1: samples
    """
    def __init__(self, shotnr, t_params, 
                 datapath="/projects/EKOLEMEN/d3dloader", device="cpu",
                 channels=['r0','v1','v2','v3']):
        """
        Unique part of constructor is channels. For CO2, it must be a subset (or default is all)
        of the 4 interferometers: r0, v1, v2, and v3
        """
        self.channels = channels
        self.name = 'raw co2 dp'
        super().__init__(shotnr, t_params, datapath, device)
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples. dim1: features (channels)
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_co2_dp.h5")) 
        tb = fp["co2_time"][:] # Get time-base
        
        t_inds = self._get_time_sampling(tb)

        # Load and stack CO2 channels, slicing happens in for loop to avoid loading data that would then be cut
        prof_data = torch.tensor(np.stack([fp["dp1"+channel+"uf"]
                                           for channel in self.channels],
                                        axis=1)[t_inds,:] 
                                    )
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading raw CO2 dp, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        # NOTE: unsqueeze(1) not needed even if there's only 1 channel 
        return prof_data


class signal_co2_pl(signal_2d):
    """Raw CO2 signals (integrated change in CO2 phase data)
    NOTE: Not really sure what this is but this is no the same as dp signal.
    It also may not be available real-time

    Returns
    -------
    co2_pl_data : tensor
                Data time series for profiles. dim0: CO2 channels. dim1: samples
    """
    def __init__(self, shotnr, t_params, 
                 datapath="/projects/EKOLEMEN/d3dloader", device="cpu",
                 channels=['r0','v1','v2','v3']):
        """
        Unique part of constructor is channels. For CO2, it must be a subset (or default is all)
        of the 4 interferometers: r0, v1, v2, and v3
        """
        self.channels = channels
        self.name = 'raw co2 pl'
        super().__init__(shotnr, t_params, datapath, device)
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples. dim1: features (channels)
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_co2_pl.h5")) 
        tb = fp["co2_time"][:] # Get time-base
        
        t_inds = self._get_time_sampling(tb)

        # Load and stack CO2 channels, slicing happens in for loop to avoid loading data that would then be cut
        prof_data = torch.tensor(np.stack([fp["pl1"+channel+"_uf"]
                                           for channel in self.channels],
                                        axis=1)[t_inds,:] 
                                    )
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading raw CO2 pl, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        # NOTE: unsqueeze(1) not needed even if there's only 1 channel 
        return prof_data


class signal_mpi(signal_2d):
    """Raw magnetic signals

    Returns
    -------
    mpi_data : tensor
                Data time series for profiles. dim0: mpi66m angles. dim1: samples
    """
    def __init__(self, shotnr, t_params, 
                 datapath="/projects/EKOLEMEN/d3dloader", device="cpu",
                 angles=[67,97,127,157,247,277,307,340]):
        """
        Unique part of constructor is angles. Must be a subset (or the default all) 
        of the following angles: 
        """
        self.angles = angles
        self.name = 'raw ece'
        super().__init__(shotnr, t_params, datapath, device)
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples. dim1: features (channels)
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        
        fp = h5py.File(join(self.datapath, "template", f"{self.shotnr}_mpi.h5")) 
        tb = fp["times"][:] # Get time-base
        
        t_inds = self._get_time_sampling(tb)

        # Load and stack MPI angles, slicing happens in for loop to avoid loading data that would then be cut
        prof_data = torch.tensor(np.stack([fp[f"mpi66m{angle:03d}f"] 
                                           for angle in self.angles],
                                        axis=1)[t_inds,:]
                                    )
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading raw MPI, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        # NOTE: unsqueeze(1) not needed even if there's only 1 channel 
        return prof_data


class signal_BES(signal_2d):
    """Raw BES signals

    Returns
    -------
    bes_data : tensor
                Data time series for profiles. dim0: BES channels. dim1: samples
    """
    def __init__(self, shotnr, t_params, 
                 datapath="/projects/EKOLEMEN/d3dloader", device="cpu",
                 channels=range(1,65)):
        """
        Unique part of constructor is channels. Can be any list of numbers from 1-64, or 
        just an individual channel. 
        """
        self.channels = channels
        self.name = 'raw BES'
        super().__init__(shotnr, t_params, datapath, device)
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Assumes the xdata and zdata keys are present and data resides in the shotnr_profile.h5 files.
        Other signals will need to override this function to cache data correctly.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples. dim1: features (channels)
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        
        fp = h5py.File(join(self.datapath, f"{self.shotnr}_BES.h5")) 
        tb = fp["times"][:] # Get time-base
        
        t_inds = self._get_time_sampling(tb)

        # Load and stack BES channels, slicing happens in for loop to avoid loading data that would then be cut
        prof_data = torch.tensor(np.stack([fp[f"BESFU{channel:02d}"]
                                           for channel in self.channels],
                                        axis=1)[t_inds,:] 
                                    )
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading raw BES, t={self.tstart}-{self.tend}s took {elapsed}s")
        
        # NOTE: unsqueeze(1) not needed even if there's only 1 channel 
        return prof_data


class signal_uci_label(signal_2d):
    """ Bill's AE labels. These are approximate as he only labeled a single time, 
    so we take assume AE mode happens in window +-250ms around labeled time. 

    Returns
    -------
    uci_label_data : tensor
                Data time series for profiles. dim0: AE modes. dim1: samples
                Order of AE modes is: BAAE, BAE, EAE, RSAE, TAE
    """
    def __init__(self, shotnr, t_params, 
                 datapath="/projects/EKOLEMEN/d3dloader", device="cpu"):
        
        self.name = 'UCI approximate AE labels'
        super().__init__(shotnr, t_params, datapath, device)
        
    def _cache_data(self):
        """Load 2d profile from hdf5 data file.
        
        Returns
        -------
        prof_data : tensor
                    Data time series for profiles. dim0: samples. dim1: features (channels)
        """

        t0_p = time.time()
        # Don't use with... scope. This throws off data_loader when running in threaded dataloader
        
        fp = h5py.File(join(self.datapath, f"{self.shotnr}_uci_label.h5")) 
        tb = fp["times"][:] # Get time-base
        
        t_inds = self._get_time_sampling(tb)

        # Load AE modes and slice time array
        prof_data = torch.tensor(fp['label'][t_inds,:])
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Loading UCI AE labels, t={self.tstart}-{self.tend}s took {elapsed}s")
        
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

