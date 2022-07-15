#!/usr/bin/env python
from os.path import join

import matplotlib.pyplot as plt

import numpy as np
import torch

import logging

logging.basicConfig(filename="d3d_loader.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

import sys
sys.path.append("/home/rkube/repos/d3d_loaders")

#from d3d_loaders.compile_dframes import process_df_bill
from d3d_loaders.d3d_loaders import D3D_dataset

t_params = {
    "tstart" : 0.001,
    "tend"   : 4000.0,
    "tsample": 1.0   
}

my_ds = D3D_dataset(169113, t_params)

# Plot the normalized Alfven eigenmode probabilities
plt.figure()
plt.plot(my_ds.ae_probs[:, 0])
plt.plot(my_ds.ae_probs[:, 1])
plt.plot(my_ds.ae_probs[:, 2])
plt.plot(my_ds.ae_probs[:, 3])
plt.plot(my_ds.ae_probs[:, 4])

# Plot the change in Alfven eigenmode probability
plt.plot(my_ds.ae_probs_delta[:, 0])
plt.plot(my_ds.ae_probs_delta[:, 1])
plt.plot(my_ds.ae_probs_delta[:, 2])
plt.plot(my_ds.ae_probs_delta[:, 3])
plt.plot(my_ds.ae_probs_delta[:, 4])
