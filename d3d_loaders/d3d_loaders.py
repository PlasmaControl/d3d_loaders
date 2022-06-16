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

class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, shotnr, tstart, tend, tsample, datapath="/projects/EKOLEMEN/aza_lenny_data1"):
        """Initializes the dataloader for DIII-D data.

        Input:
        =======
        shotnr...: (int) shot number
        tstart...: (float) Start time, in milliseconds
        tend.....: (float) End time, in milliseconds
        subsample..: (float) Time between samples, in milliseconds

        Timebase of
        * ECE: dt_ece = 2e-3 ms
        * Pinj: dt_p = 1e-2 ms
        * Neutrons: dt_neu = 2e-2ms
i
        At each iteration, the data loader fetches a set of data for a given 
        shot for
        * t0
        * t0 + 50μs.

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

        assert(tstart < tend)
        assert((tend - tstart) > tsample)

        # Load RCN weights
        with open('/home/rkube/ml4control/1dsignal-model-AE-ECE-RCN.pkl', 'rb') as df:
            self.infer_data = pickle.load(df)
        self.n_res_l1 = self.infer_data['layer1']['w_in'].shape[0]
        self.n_res_l2 = self.infer_data['layer2']['w_in'].shape[0]

        # Caching: Calculate all AE mode probabilities and load pinj, neutron data
        self.ae_probs = self._cache_ae_prob()
        self.data_pinj = self._cache_data_pinj()
        self.data_neut = self._cache_data_neut()
    
        # AE mode prob delta is the change in mode probabilities
        self.ae_probs_delta = self.ae_probs[:, :, 1] - self.ae_probs[:, :, 0]

        # Z-Score normalization, store mean and std as member vars
        self.ae_probs_mean = self.ae_probs[:, :, 0].mean()
        self.ae_probs_std = self.ae_probs[:, :, 0].std()
        self.ae_probs = (self.ae_probs[:, :, 0] - self.ae_probs_mean) / self.ae_probs_std

        self.ae_probs_delta_mean = self.ae_probs_delta.mean()
        self.ae_probs_delta_std = self.ae_probs_delta.std()
        self.ae_probs_delta = (self.ae_probs_delta - self.ae_probs_delta_mean) / self.ae_probs_delta_std

        self.data_pinj_mean = self.data_pinj.mean()
        self.data_pinj_std = self.data_pinj.std()
        self.data_pinj = (self.data_pinj - self.data_pinj_mean) / self.data_pinj_std

        self.data_neut_mean = self.data_neut.mean()
        self.data_neut_std = self.data_neut.std()
        self.data_neut = (self.data_neut - self.data_neut_mean) / self.data_neut_std



    def __len__(self):
        return len(self.target_df)


    def _get_num_n_samples(self, dt):
        """Calculate number of samples and sample skipping.

        Given a signal sampled on [tstart:tend] with sample spacing dt
        calculate the total number of samples available.

        Also calculates the sample skipping when sub-sampling dt to self.tsample.
        """

        num_samples = int(ceil((self.tend - self.tstart) / dt))
        nth_sample = int(ceil(self.tsample / dt))

        return (num_samples, nth_sample)


    def _cache_ae_prob(self):
        """Forward pass for RCN model that predicts AE mode probabilities.

        This has to be done sequential. Iterative over all samples."""

        # Find how many samples apart tsample is
        t0_p = time.time()
        with h5py.File(join(self.datapath, "template", f"{self.shotnr}_ece.h5"), "r") as fp:
            tb_ece = fp["ece"]["xdata"][:]    # Get ECE time-base
            dt_ece = np.diff(tb_ece).mean()   # Get ECE sampling time
            num_samples, nth_sample = self._get_num_n_samples(dt_ece)
            t0_idx = np.argmin(np.abs(tb_ece - self.tstart))
            logging.info(f"Sampling ECE: t0_idx={t0_idx}, dt={dt_ece}, num_samples={num_samples}, nth_sample={nth_sample}")

            # Read in all ece_data at t0 and shifted at t0 + 50 mus
            ece_data_0 = np.vstack([fp["ece"][f"tecef{(i+1):02d}"][t0_idx:t0_idx + num_samples:nth_sample ] for i in range(40)]).T
            ece_data_1 = np.vstack([fp["ece"][f"tecef{(i+1):02d}"][t0_idx + 25:t0_idx + 25 + num_samples:nth_sample] for i in range(40)]).T
            # After this we have ece_data_0.shape = (num_samples / nth_sample, 40)

            # Pre-allocate array for AE mode probabilities
            # dim0: time index
            # dim1: AE mode index 0...4
            # dim2: 0: t0, 1: t0 + 50mus
            ae_probs = np.zeros([ece_data_0.shape[0], 5, 2], dtype=np.float32)

            ece_data_0 = (ece_data_0 - self.infer_data["mean"]) / self.infer_data["std"]
            ece_data_1 = (ece_data_1 - self.infer_data["mean"]) / self.infer_data["std"]
            # Initialize to zero, overwrite in for loop
            r_prev = {"layer1": np.zeros(self.n_res_l1),
                      "layer2": np.zeros(self.n_res_l2)} 
            # Iterate over time index 0
            for idx, u in enumerate(ece_data_0):
                L = "layer1"
                r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                         self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                         self.infer_data[L]['leak_rate'], r_prev[L], u.T)

                L = "layer2"
                r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                         self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                         self.infer_data[L]['leak_rate'], r_prev[L], y)

                ae_probs[idx, :, 0] = y[:]

            # Re-set weights to zero and overwrite again in for-loop
            r_prev = {"layer1": np.zeros(self.n_res_l1),
                      "layer2": np.zeros(self.n_res_l2)} 
            for idx, u in enumerate(ece_data_1):
                L = "layer1"
                r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                         self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                         self.infer_data[L]['leak_rate'], r_prev[L], u.T)

                L = "layer2"
                r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                         self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                         self.infer_data[L]['leak_rate'], r_prev[L], y)

                ae_probs[idx, :, 1] = y[:]
        elapsed = time.time() - t0_p

        #if logger is not None:
        logging.info(f"AE forward model for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")

        return torch.tensor(ae_probs)

   
    def _cache_data_pinj(self):
        """Loads sum of all pinj at t0 and t0+50ms"""

        # Calculate samples for pinj data
        # Load pinj data at t0 and t0 + 50ms. dt for this data is 10ms
        t0_p = time.time()
        with h5py.File(join(self.datapath, "template", f"{self.shotnr}_pinj.h5")) as fp_pinj:
            tb_pinj = torch.tensor(fp_pinj["pinjf_15l"]["xdata"][:]) # Get time-base
            dt_pinj = np.diff(tb_pinj).mean()                        # Get sampling time
            dt_pinj = 0.01
            # Get total number of samples and desired sub-sample spacing
            num_samples, nth_sample = self._get_num_n_samples(dt_pinj)
            t0_idx = torch.argmin(torch.abs(tb_pinj - self.tstart))
            logging.info(f"Sampling pinj: t0_idx={t0_idx}, dt={dt_pinj}, num_samples={num_samples}, nth_sample={nth_sample}")

            pinj_data_0 = sum([torch.tensor(fp_pinj[k]["zdata"][:])[t0_idx:t0_idx + num_samples:nth_sample] for k in fp_pinj.keys()])
            pinj_data_1 = sum([torch.tensor(fp_pinj[k]["zdata"][:])[t0_idx + 5:t0_idx + num_samples + 5:nth_sample] for k in fp_pinj.keys()])

        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading pinj for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
            
        return torch.stack((pinj_data_0, pinj_data_1))
    
    def _cache_data_neut(self):
        """Loads neutron emission rate at t0 and t0+50mus"""
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        with h5py.File(join(self.datapath, "template", f"{self.shotnr}_profiles.h5")) as fp_prof:

            tb_neu = torch.tensor(fp_prof["neutronsrate"]["xdata"][:])
            dt_neu = np.diff(tb_neu).mean()
            dt_neu = 0.02
            num_samples, nth_sample = self._get_num_n_samples(dt_neu)
            t0_idx = torch.argmin(torch.abs(tb_neu - self.tstart))
            logging.info(f"Sampling neut: t0_idx={t0_idx}, dt={dt_neu}, num_samples={num_samples}, nth_sample={nth_sample}")

            neutron_data_0 = torch.tensor(fp_prof["neutronsrate"]["zdata"][t0_idx:t0_idx + num_samples:nth_sample])
            neutron_data_1 = torch.tensor(fp_prof["neutronsrate"]["zdata"][t0_idx + 1:t0_idx + num_samples:nth_sample])
            
        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading neutron data for {self.shotnr}, t={self.tstart}-{self.tend}s took {elapsed}s")
        return torch.stack((neutron_data_0, neutron_data_1))
    

    def __getitem__(self, idx):
        """Fetch data corresponding to the idx'th sample."""
        data_t0 = torch.cat((self.ae_probs[idx, :], 
                             self.data_neut[0, idx].unsqueeze(0), 
                             self.data_pinj[0, idx].unsqueeze(0)))
        data_t1 = self.ae_probs_delta[idx, :]
        return (data_t0, data_t1)
#
# End of file d3d_loaders.py
