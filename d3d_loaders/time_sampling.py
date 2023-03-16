#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
    Defines samplers that sample signals onto the same time-base
"""

import torch
import numpy as np
import logging
from scipy.interpolate import interp1d

class sampler_base:
    """Resamples signals on given time base

    The arguments define a time base: range(t_start, t_end, dt) - t_shift.

    Args:
        t_start (float) : Start time of resampling time-base (in milliseconds)
        t_end (float) : End time of resampling time-base (in milliseconds)
        dt (float) : Sample spacing (in milliseconds)
        t_shift (float) : Shift entire time series by t_shift into the future (in milliseconds)

    """
    def __init__(self, t_start, t_end, dt, t_shift=0.0):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.t_shift = t_shift

    def get_sample_indices(self, tb):
        """Does nothing."""
        return tb


class sampler_causal(sampler_base):
    """Causal re-sampler. """

    def __init__(self, t_start, t_end, dt, t_shift=0.0):
        super(sampler_causal, self).__init__(t_start, t_end, dt, t_shift)

    def get_sample_indices(self, tb):
        """Picks time indices to causally resample signal on desired timebase

        Calculates indices to causally map a signal sampled at the timebase defined in `tb`
        onto the timebase defined by range(t_start, t_end, dt)
        Args:
            tb (torch.tensor) : Time-base of input signal

        Returns:
            t_inds (torch.tensor) : Indices for desired sampling
        """
        # Raise error if first time sample comes before first measurement
        if tb[0] > self.t_start:
            raise(ValueError(f'Time of first requested sample is before first real measurement was taken'))

        # Force data onto these sample times
        time_sample_vals = torch.arange(self.t_start, self.t_end, self.dt)

        # Shift time series.
        time_sample_vals += self.t_shift
        logging.info(f"Shifting time_sample_vals = {self.t_shift}ms")
        # Number of new samples to generate
        num_samples = time_sample_vals.numel()
        t_inds = torch.zeros((num_samples), dtype=torch.int)

        # Iterate over array and find find index in the input time base that fills
        # the sample in the new time base
        tb_ind = 1
        for i, time_samp in enumerate(time_sample_vals):
            # Scan time base as long as we are below the next desired sampling time
            while tb[tb_ind] < time_samp and tb_ind < tb.numel() - 1:
                tb_ind += 1

            t_inds[i] = tb_ind - 1

        return t_inds

    def __repr__(self):
        return f"causal_sampler: ({self.t_start}, {self.t_end}, {self.dt}), shift={self.t_shift}ms"


class sampler_linearip(sampler_base):
    """Linear interpolatoin sampler."""
    def __init__(self, t_start, t_end, dt, t_shift=0.0):
        super(sampler_linearip, self).__init__(t_start, t_end, dt, t_shift)
        self.tb_new = np.arange(t_start - t_shift, t_end - t_shift, dt)

    def resample(self, tb, signal, dtype=np.float32):
        """Interpolate signal on new timebase."""
        # Convert to numpy so we got interpolate.interp1d working
        if isinstance(signal, torch.Tensor):
            signal = signal.numpy()

        if isinstance(tb, torch.Tensor):
            tb = tb.numpy()

        f = interp1d(tb, signal)
        sig_rs = f(self.tb_new)
        # Somewhere in this routine, float32 (input) gets converted to float64.
        # Enforce output type here.
        return sig_rs.astype(dtype)

    def __repr__(self):
        return f"sampler_linearip: {self.t_start}, {self.t_end}, {self.dt}), shift={self.t_shift}ms"



class sampler_space():
    """Spatial resampling for profiles."""
    def __init__(self, x_new):
        """Set up interpolation to new spatial basis.

        This method just checks if the new points are sorted and stores them.        
        Args:
           
            x_new (ndarray): Collocation points of new profile
        """
        # Test if new collocation points are sorted
        assert(((x_new[1:] - x_new[:-1]) > 0.0).all())
        self.x_new = np.array(x_new)
        
    def __call__(self, x_old, prof_old):
        """Interpolate array from old to new collocation points.
        
        Args:
            x_old (np.ndarray) : Collocation points of the profile
            prof_old: Profile values on old collocation points. Dim0: spatial points. Dim1: Time dimension
            
        Returns:
            prof_new: Profile values at new collocation points.
        """
        # Test if both, old and new collocation points are sorted
        assert(((x_old[1:] - x_old[:-1]) > 0.0).all())
        # Assert that new collocations points are within original bounds
        # No extrapolation
        assert((self.x_new[0] - x_old[0]).item() >= 0.0)
        assert((self.x_new[-1] - x_old[-1]).item() <= 0.0)
        x_old = np.array(x_old)

        # Interpolate profile on x_new and return.
        ip = interp1d(x_old, prof_old, axis=1)
        prof_new = ip(self.x_new)
        
        return prof_new
        

# end of file time_sampling.py