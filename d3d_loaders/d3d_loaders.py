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

import pickle
from rcn_functions import rcn_infer

class D3D_dataset(torch.utils.data.Dataset):
    """Implements an iterable dataset for D3D data.

    Target is the HDF5 data stored in /projects/EKOLEMEN/aza_lenny_data1.
    
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    def __init__(self, datapath="/projects/EKOLEMEN/aza_lenny_data1"):
        """Initializes the dataloader. 

        At each iteration, the data loader fetches a set of data for a given 
        shot for
        * t0
        * t0 + 50μs.

        Built in to this dataset is a set of (shot, t0). These are defined by
        the intersection of the shots and time points defined in `uci_ae_labels.txt`
        and those shots for which neutronrate data is available.
        """

        super(D3D_dataset).__init__()
        from d3d_loaders.rcn_functions import preprocess_label
        # Directory where ECE, Profiles, and Pnbi data are stored
        self.datapath = datapath

        # Populate the dataframe for ECE data. This provides the first set of (shots, t0)
        self.ece_label_df = preprocess_label(join(self.datapath,  "uci_ae_labels.txt"))
        
        # Load shots for which neutron rates are available. This provides the second set of (shots,)
        shots_neutrons = []
        with open(join(datapath, "shots_neutrons.txt"), "r") as f:
            for line in f.readlines():
                shots_neutrons.append(int(line))       
        
        # Now form the intersection of both sets of shots.
        # The result is a list of shots which we have ece labels and neutron rates
        self.shot_list = list(set(shots_neutrons).intersection(set(self.ece_label_df.shot.tolist())))
        self.shot_list.sort()
        # Remove all rows for which there is no neutron data
        self.ece_label_df = self.ece_label_df[self.ece_label_df["shot"].isin(self.shot_list)]

        # Load RCN weights
        with open('/home/rkube/ml4control/1dsignal-model-AE-ECE-RCN.pkl', 'rb') as df:
            self.infer_data = pickle.load(df)
        self.n_res_l1 = self.infer_data['layer1']['w_in'].shape[0]
        self.n_res_l2 = self.infer_data['layer2']['w_in'].shape[0]
    

    def __len__(self):
        return len(self.ece_label_df)


    def fetch_data_ece(self, shotnr, t0):
        """Loads ECE at t0 and t0+50μs"""
        
        # Load ECE data at t0 and t0 + 50mus. dt for this data is 2mus
        with h5py.File(join(self.datapath, "template", f"{shotnr}_ece.h5"), "r") as fp:
            ece_t0_idx = torch.squeeze(torch.argwhere(torch.tensor(fp["ece"]["xdata"][:])  < t0))[-1]
            ece_data_0 = torch.vstack([torch.tensor(fp["ece"][f"tecef{(i+1):02d}"][ece_t0_idx]) for i in range(40)]).T
            ece_data_1 = torch.vstack([torch.tensor(fp["ece"][f"tecef{(i+1):02d}"][ece_t0_idx + 25]) for i in range(40)]).T
            
        return (ece_data_0, ece_data_1)
    
    def fetch_data_pinj(self, shotnr, t0):
        """Loads sum of all pinj at t0 and t0+50ms"""
        # Load pinj data at t0 and t0 + 50ms. dt for this data is 10ms
        with h5py.File(join(self.datapath, "template", f"{shotnr}_pinj.h5")) as df_pinj:
            xdata = torch.tensor(df_pinj["pinjf_15l"]["xdata"][:])
            pinj_t0_idx = torch.squeeze(torch.argwhere(xdata  < t0))[-1]
            pinj_data_0 = sum([torch.tensor(df_pinj[k]["zdata"][:])[pinj_t0_idx] for k in df_pinj.keys()])
            pinj_data_1 = sum([torch.tensor(df_pinj[k]["zdata"][:])[pinj_t0_idx + 5] for k in df_pinj.keys()])
            
        return (pinj_data_0, pinj_data_1)
    
    def fetch_data_neu(self, shotnr, t0):
        """Loads neutron emission rate at t0 and t0+50mus"""
        # Load neutron data at t0 and t0 + 50ms. dt for this data is 50ms
        with h5py.File(join(self.datapath, "template", f"{shotnr}_profiles.h5")) as df_prof:
            xdata = torch.tensor(df_prof["neutronsrate"]["xdata"][:])
            neu_t0_idx = torch.squeeze(torch.argwhere(xdata  < t0))[-1]
            neutron_data_0 = torch.tensor(df_prof["neutronsrate"]["zdata"][neu_t0_idx])
            neutron_data_1 = torch.tensor(df_prof["neutronsrate"]["zdata"][neu_t0_idx + 1])
            
        return (neutron_data_0, neutron_data_1)
    
    def ae_mode_prob(self, shotnr, t0):
        """Forward pass for RCN model that predicts AE mode probabilities."""
        ece_data_0, ece_data_1 = self.fetch_data_ece(shotnr, t0)
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

        return ae_mode_probs


    def __getitem__(self, idx):
        shotnr, t0 = self.ece_label_df[["shot", "time"]].iloc[idx]

        ae_mode_prob_t0, ae_mode_prob_t1 = self.ae_mode_prob(shotnr, t0)
        pinj_t0, pinj_t1 = self.fetch_data_pinj(shotnr, t0)
        nrate_t0, nrate_t1 = self.fetch_data_neu(shotnr, t0)

        return (ae_mode_prob_t0, pinj_t0, nrate_t0), (ae_mode_prob_t1, pinj_t1, nrate_t1)


#        """Implements iteration over valid shots."""
#        worker_info = torch.utils.data.get_worker_info()
#        if worker_info is None: # single-process loading, return the full iterator
#            iter_start = self.start
#            iter_end = self.end
#        else: # in a worker process
#            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#            worker_id = worker_info.id
#            iter_start = self.start + worker_id * per_worker
#            iter_end = min(iter_start + per_worker, self.end)
#            
#        # start and end both refer to a row in the event dataframe.
#        # Each event consists of a shot and time-in-shot
#        return iter((self.fetch_data_ece(shotnr, t), 
#                     self.fetch_data_pinj(shotnr, t),
#                     self.fetch_data_neu(shotnr, t)) for shotnr, t in zip(self.ece_label_df[["shot"]].iloc[iter_start:iter_end].shot,
                                                                            #self.ece_label_df[["time"]].iloc[iter_start:iter_end].time))
