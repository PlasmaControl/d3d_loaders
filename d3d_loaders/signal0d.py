#-*- encoding: utf-8 -*-

"""Contains class definitions used to as abstractions for 1d signals."""

from os.path import join
import time
import logging

import numpy as np
import h5py
import torch
import yaml

import importlib.resources

import d3d_signals


class signal_base():
    """Base class for 0d (scalar) signals. """
    def __init__(self, shotnr, time_sampler, standardizer, datapath, device=torch.device("cpu")):
        """Load data from HDF5 file, standardize, and move to device.

        Parameters
        ----------
        shotnr : int
                 Shot number
        time_sampler : class `causal_sampler`. 
                Defines the timebase on which we need the signal          
        standardizer : class `standardizer`
                Defines standardization of the dataset.         
        datapath : string, default='/projects/EKOLEMEN/aza_lenny_data1'
                   Basepath where HDF5 data is stored.  
        device : string, default='cpu'
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        # Store function arguments as member variables
        self.shotnr = shotnr
        self.time_sampler = time_sampler
        self.standardizer = standardizer
        self.datapath = datapath
        
        # Load data from HDF5 file and store, move to device
        data = self._cache_data().to(device)
        self.data = self.standardizer(data)
        logging.info(f"""Compiled signal {self.__class__.__name__}, 
                         resampler={self.time_sampler},
                         standardizer={self.standardizer},
                         datapath={self.datapath}""")

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
        t0_p = time.time()
        # Don't use with... scope. This throws off dataloader
        fp = h5py.File(join(self.datapath, f"{self.shotnr}.h5"), "r") 
        # Checks to make sure predictor is present
        try:
            tb = torch.tensor(fp[self.key]["xdata"][:])
        except ValueError as e:
            fp.close()
            logging.error(f"Unable to load timebase for shot {self.shotnr} signal {self.name}")
            raise e

        # Some shots have no data for a given signal. In that case, the tensor is present in the
        # dataset but the size is 0. Throw an error if that is the case.
        if tb.shape[0] < 2:
            fp.close()
            raise ValueError(f"Shot {self.shotnr}, signal {self.key}: Timebase in HDF5 file has length {tb.shape[0]} < 2!")
        
        # Indices to sample on. Translate to numpy array
        t_inds = self.time_sampler.get_sample_indices(tb).numpy()
        data = torch.tensor(fp[self.key]["zdata"][:])[t_inds]
        fp.close()

        elapsed = time.time() - t0_p       
        logging.info(f"Caching {self.name} data for {self.shotnr} took {elapsed}s")

        return data.unsqueeze(1)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"signal_base({self.data}"


class signal_pinj(signal_base):
    """Sum of total injected power.
    
        Args:
        -----
            name (str):
            collect_keys (list(str))): List of datasets in the HDF5 file that will be summed to build the output signal.
        

        This signal is constructed by summing over a list of neutral beam injected power time
        series. The list of signal over which we sum is given by the collect_keys argument.
    """
    def __init__(self, shotnr, time_sampler, std, datapath, device):
        self.name = 'pinj'
        self.collect_keys = ["pinjf_15l", "pinjf_15r", "pinjf_21l", "pinjf_21r", 
                             "pinjf_30l", "pinjf_30r", "pinjf_33l", "pinjf_33r"]
        super().__init__(shotnr, time_sampler, datapath, device)
    
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
        t_inds = self.time_sampler.get_sample_indices(tb).numpy()
        # Sum the contributions from all neutral beams specified in the collect_keys list.
        pinj_data = sum([torch.tensor(fp[k]["zdata"][:])[t_inds] for k in self.collect_keys])
        fp.close()

        elapsed = time.time() - t0_p
        logging.info(f"Caching pinj data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
           
        return pinj_data.unsqueeze(1)


def signal_factory(full_name):
    """Create a signal for arbitrary names.
    
    Args:
        full_name (str) : Name of the signal. Needs to begin with "signal_"
        base_class (type) : Needs 
    
        
    This function returns a dynamically generated sub-class of signal_base.
    
    It follows the design principle that is called object factory. 


    See https://stackoverflow.com/questions/15247075/how-can-i-dynamically-create-derived-classes-from-a-base-class

    This works like this:
    >>> signal_efsli = signal_factory("signal_efsli")
    >>> efsli = signal_efsli(shotnr, sampler_causal(100.0, 1000.0, 1.0), std, datapath, "cpu")
    >>> plt.plot(np.arange(100.0, 1000.0, 1.0), efsli.data[:,0].numpy())

    Here, signal_efsli is the definition of a class. The code above roughly does the same as
    1. defining the class in code

    class signal_efsli():
        def __init___(self, ...)
    ...
    2. and the instantiating it.
    efsli = signal_efsli(shotnr, sampler_causal...)

    This function dynamically generates the class definition for signal_efsli.


    When dispatching to HDF5, the signal class will use self.key to access the relevant 
    data group. This key is taken from the signal definition signals_0d.yaml

    This factory needs to be able to import d3d_signals. Make sure you have the module
    https://github.com/PlasmaControl/d3d_signals
    checked out and are able to import it, that is,
    >>> import d3d_loaders
    should work.

    """
    assert(full_name[:7] == "signal_")
    short_name = full_name[7:]   # The part after signal_

    # Access signal definition from yaml files: 
    # s = Sour
    # Use importlib to guarantee path safety
    try:
        resource_path = importlib.resources.files("d3d_signals")
    except ModuleNotFoundError as e:
        print("Could not load submodule 'd3d_signals'")
        print("Please manually import the subfolder 'd3d_signals' in the script you are running from")
        print("")
        print(">>> import sys")
        print(">>> sys.path.append(/path/to/d3d_signals)")
        print(">>> import d3d_signals")

        raise e

    with open(join(resource_path, "signals_0d.yaml"), "r") as fp:
        signals_0d = yaml.safe_load(fp)
   
    # Define __init__ function for new signal
    def __init__(self, shotnr, time_sampler, std, datapath, device):
        self.name = short_name
        self.key = signals_0d[short_name]["map_to"]
        
        signal_base.__init__(self, shotnr, time_sampler, std, datapath, device=torch.device("cpu"))
        
    # The magic here is that newclass is a !!!type!!!. In particular, newclass is not
    # an instance of any class. But a type. A type like Int, or Float, or str.
    #
    # a = "foobar"  <---- a is an instance of str.
    # b = 9         <---- b is an instance of int.
    # c = type(b)   <---- c is an instance of a type.
    # Now we can do things like
    # d = c(9)      <---- d is an integer with value 9
    # d + 12        <---- since d is an integer, we can add it to another integer.
    # 21      
    newclass = type(full_name, (signal_base, ), {"__init__": __init__})
    # Now we just return the object.
    return newclass

