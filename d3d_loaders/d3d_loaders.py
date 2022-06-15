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

import pickle
from rcn_functions import rcn_infer

import logging

class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, target_df, datapath="/projects/EKOLEMEN/aza_lenny_data1"):
        """Initializes the dataloader. 

        At each iteration, the data loader fetches a set of data for a given 
        shot for
        * t0
        * t0 + 50μs.

        The shot number and t0's are given by columns "shot" and "time in target_df.

        Built in to this dataset is a set of (shot, t0). These are defined by
        the intersection of the shots and time points defined in `uci_ae_labels.txt`
        and those shots for which neutronrate data is available.
        """

        super(D3D_dataset).__init__()
        # Directory where ECE, Profiles, and Pnbi data are stored
        self.datapath = datapath
        self.target_df = target_df

        # Load RCN weights
        with open('/home/rkube/ml4control/1dsignal-model-AE-ECE-RCN.pkl', 'rb') as df:
            self.infer_data = pickle.load(df)
        self.n_res_l1 = self.infer_data['layer1']['w_in'].shape[0]
        self.n_res_l2 = self.infer_data['layer2']['w_in'].shape[0]
    

    def __len__(self):
        return len(self.target_df)


    def fetch_data_ece(self, shotnr, t0):
        """Loads ECE at t0 and t0+50μs"""
        
        # Load ECE data at t0 and t0 + 50mus. dt for this data is 2mus
        t0_p = time.time()
        with h5py.File(join(self.datapath, "template", f"{shotnr}_ece.h5"), "r") as fp:
            ece_t0_idx = torch.squeeze(torch.argwhere(torch.tensor(fp["ece"]["xdata"][:])  < t0))[-1]
            ece_data_0 = torch.vstack([torch.tensor(fp["ece"][f"tecef{(i+1):02d}"][ece_t0_idx]) for i in range(40)]).T
            ece_data_1 = torch.vstack([torch.tensor(fp["ece"][f"tecef{(i+1):02d}"][ece_t0_idx + 25]) for i in range(40)]).T
        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading ECE data for {shotnr}, t={t0} took {elapsed}s")
            
        return (ece_data_0, ece_data_1)
    
    def fetch_data_pinj(self, shotnr, t0):
        """Loads sum of all pinj at t0 and t0+50ms"""
        # Load pinj data at t0 and t0 + 50ms. dt for this data is 10ms
        t0_p = time.time()
        with h5py.File(join(self.datapath, "template", f"{shotnr}_pinj.h5")) as df_pinj:
            xdata = torch.tensor(df_pinj["pinjf_15l"]["xdata"][:])
            pinj_t0_idx = torch.squeeze(torch.argwhere(xdata  < t0))[-1]
            pinj_data_0 = sum([torch.tensor(df_pinj[k]["zdata"][:])[pinj_t0_idx] for k in df_pinj.keys()])
            pinj_data_1 = sum([torch.tensor(df_pinj[k]["zdata"][:])[pinj_t0_idx + 5] for k in df_pinj.keys()])

        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading Pinj data for {shotnr}, t={t0} took {elapsed}s")
            
        return (pinj_data_0, pinj_data_1)
    
    def fetch_data_neu(self, shotnr, t0):
        """Loads neutron emission rate at t0 and t0+50mus"""
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        with h5py.File(join(self.datapath, "template", f"{shotnr}_profiles.h5")) as df_prof:
            xdata = torch.tensor(df_prof["neutronsrate"]["xdata"][:])
            neu_t0_idx = torch.squeeze(torch.argwhere(xdata  < t0))[-1]
            neutron_data_0 = torch.tensor(df_prof["neutronsrate"]["zdata"][neu_t0_idx])
            neutron_data_1 = torch.tensor(df_prof["neutronsrate"]["zdata"][neu_t0_idx + 1])
            
        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading neutron rate data for {shotnr}, t={t0} took {elapsed}s")
        return (neutron_data_0, neutron_data_1)
    
    def ae_mode_prob(self, shotnr, t0):
        """Forward pass for RCN model that predicts AE mode probabilities."""
        ece_data_0, ece_data_1 = self.fetch_data_ece(shotnr, t0)

        t0_p = time.time()
        ece_data_0 = np.squeeze(ece_data_0.numpy())
        ece_data_1 = np.squeeze(ece_data_1.numpy())
        r_prev = {"layer1": np.zeros(self.n_res_l1),
                  "layer2": np.zeros(self.n_res_l2)} 

        ae_mode_probs = []
        for u in [ece_data_0, ece_data_1]:
            u = (u - self.infer_data['mean']) / self.infer_data['std'] #Normalizing the input
            L = "layer1"
            r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                     self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                     self.infer_data[L]['leak_rate'], r_prev[L], u.T)

            L = "layer2"
            r_prev[L], y = rcn_infer(self.infer_data[L]['w_in'], self.infer_data[L]['w_res'],
                                     self.infer_data[L]['w_bi'], self.infer_data[L]['w_out'],
                                     self.infer_data[L]['leak_rate'], r_prev[L], y)
            ae_mode_probs.append(y)
        ae_mode_probs = np.array(ae_mode_probs, dtype=np.float32)
        elapsed = time.time() - t0_p

        #if logger is not None:
        logging.info(f"AE forward model for {shotnr}, t={t0} took {elapsed}s")

        return ae_mode_probs


    def __getitem__(self, idx):
        shotnr, t0 = self.target_df[["shot", "time"]].iloc[idx]
        shotnr = int(shotnr)
        t0 = float(t0)
        print(f"Loading data for {shotnr} t={t0}")

        ae_mode_prob_t0, ae_mode_prob_t1 = self.ae_mode_prob(shotnr, t0)
        pinj_t0, pinj_t1 = self.fetch_data_pinj(shotnr, t0)
        nrate_t0, nrate_t1 = self.fetch_data_neu(shotnr, t0)

        return (ae_mode_prob_t0, pinj_t0, nrate_t0), (ae_mode_prob_t1, pinj_t1, nrate_t1)


class D3D_dataset_scoped(D3D_dataset):
    """D3D_dataset with caching of open HDF5 files."""
    def __init__(self, shot_df, datapath="/projects/EKOLEMEN/aza_lenny_data1"):
        super().__init__(shot_df, datapath)


    def __enter__(self):
        """Cache all hdf5 file handles."""
        # All file pointers to open HDF5 files are cached in a dictionary
        # One dictionary per diagnostic: ECE, neutron rate, Pinj
        # Keys: shot_number     Value: File pointer
        print("Initializing file handle dictionaries")
        self.fp_ece_dict = {}
        self.fp_neu_dict = {}
        self.fp_pin_dict = {}

        # Need to return self here:
        # https://stackoverflow.com/questions/5093382/object-becomes-none-when-using-a-context-manager
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close all cached file pointers when exiting scope"""

        print("Clearing file-handle dictionaries")
        for k, fp in self.fp_ece_dict:
            print(f"Closing ece file handle for {k}")
            fp.close()

        for k, fp in self.fp_neu_dict:
            print(f"Closing neutron file handle for {k}")
            fp.close()

        for k, fp in self.fp_pin_dict:
            print(f"Closing pin file handle for {k}")
            fp.close()

    def fetch_data_ece(self, shotnr, t0):
        """Loads ECE at t0 and t0+50μs"""
        
        # Load ECE data at t0 and t0 + 50mus. dt for this data is 2mus
        t0_p = time.time()
        # Check if the file handle for the show is already open. If not, open
        # it and append to dict
        try:
            fp = self.fp_ece_dict[shotnr]
        except KeyError:
            fp = h5py.File(join(self.datapath, "template", f"{shotnr}_ece.h5"), "r") 
            self.fp_ece_dict[shotnr] = fp

        ece_t0_idx = torch.squeeze(torch.argwhere(torch.tensor(fp["ece"]["xdata"][:])  < t0))[-1]
        ece_data_0 = torch.vstack([torch.tensor(fp["ece"][f"tecef{(i+1):02d}"][ece_t0_idx]) for i in range(40)]).T
        ece_data_1 = torch.vstack([torch.tensor(fp["ece"][f"tecef{(i+1):02d}"][ece_t0_idx + 25]) for i in range(40)]).T
        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading ECE data for {shotnr}, t={t0} took {elapsed}s")
            
        return (ece_data_0, ece_data_1)
    
    def fetch_data_pinj(self, shotnr, t0):
        """Loads sum of all pinj at t0 and t0+50ms"""
        # Load pinj data at t0 and t0 + 50ms. dt for this data is 10ms
        t0_p = time.time()
        try:
            fp = self.fp_pin_dict[shotnr]
        except KeyError:
            fp = h5py.File(join(self.datapath, "template", f"{shotnr}_pinj.h5"))
            self.fp_pin_dict[shotnr] = fp

        print(fp.file)

        xdata = torch.tensor(fp["pinjf_15l"]["xdata"][:])
        pinj_t0_idx = torch.squeeze(torch.argwhere(xdata  < t0))[-1]
        pinj_data_0 = sum([torch.tensor(fp[k]["zdata"][:])[pinj_t0_idx] for k in fp.keys()])
        pinj_data_1 = sum([torch.tensor(fp[k]["zdata"][:])[pinj_t0_idx + 5] for k in fp.keys()])

        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading Pinj data for {shotnr}, t={t0} took {elapsed}s")
            
        return (pinj_data_0, pinj_data_1)
    
    def fetch_data_neu(self, shotnr, t0):
        """Loads neutron emission rate at t0 and t0+50mus"""
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        t0_p = time.time()
        try:
            fp = self.fp_neu_dict[shotnr]
        except KeyError:
            fp = h5py.File(join(self.datapath, "template", f"{shotnr}_profiles.h5"))
            self.fp_neu_dict[shotnr] = fp

        xdata = torch.tensor(fp["neutronsrate"]["xdata"][:])
        neu_t0_idx = torch.squeeze(torch.argwhere(xdata  < t0))[-1]
        neutron_data_0 = torch.tensor(fp["neutronsrate"]["zdata"][neu_t0_idx])
        neutron_data_1 = torch.tensor(fp["neutronsrate"]["zdata"][neu_t0_idx + 1])
            
        elapsed = time.time() - t0_p
        #if logger is not None:
        logging.info(f"Loading neutron rate data for {shotnr}, t={t0} took {elapsed}s")
        return (neutron_data_0, neutron_data_1)
 
# End of file d3d_loaders.py
