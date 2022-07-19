#!/usr/bin/env python

import sys
sys.path.append("/home/rkube/repos/d3d_loaders")

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import logging
from d3d_loaders.d3d_loaders import D3D_dataset

# Consider 2 seconds of shot 163163 and 196113
shotnr = [163163, 169113]

# Time sampling parameters, all in ms
t_params = {
    "tstart" : 0.001,  # Start time
    "tend"   : 2000.0, # End time
    "tsample": 1.0     # Sampling rate
}

# Which signals to use for prediction
predictors = ["ae_prob", "pinj", "neut", "ip", "dens"]

# Calculate changes in probabilities 10ms into future
shift = {'ae_prob_delta':10.0}

batch_size = 64
num_epochs = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a dataset using
# * pinj, neutrons, Alfven Eigenmode probability as predictors
# * Change in Alfven probability over 10 ms as target
# * Data will be loaded from HDF5 files and moved to the device
ds = D3D_dataset(shotnr, t_params, 
        predictors=predictors,
        targets=["ae_prob_delta"],
        shift_targets=shift,
        device=device)

# Set up train/validation split

split = 0.8
num_samples = len(ds)
num_train = int(num_samples * split)
num_val = num_samples - num_train

ds_train, ds_val = torch.utils.data.random_split(ds, [num_train, num_val])


# Define loaders for training and validation sets:
loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, num_workers=0, shuffle=True)
loader_valid = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=0, shuffle=True)

# Use a simple MLP as the model
class my_mlp(nn.Module):
    """MLP that maps final_output[0] to final_output[1]."""
    def __init__(self):
        super(my_mlp, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)

        return x

# Train the model
model = my_mlp().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()


losses_train_epoch = np.zeros(num_epochs)
losses_valid_epoch = np.zeros(num_epochs)

total_batch_tr = 0
total_batch_val = 0
for epoch in range(num_epochs):
    model.train()

    loss_train = 0.0
    loss_valid = 0.0
    for i, (data, target) in enumerate(loader_train):
        optimizer.zero_grad()

        outputs = model(data)

        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    with torch.no_grad():
        for i, (data, target) in enumerate(loader_valid):
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss_valid += loss.item()

    losses_train_epoch[epoch] = loss_train / len(loader_train) / batch_size
    losses_valid_epoch[epoch] = loss_valid / len(loader_valid) / batch_size

    if epoch % 10 == 0:
        print(f"epoch: {epoch+1:2d}, train loss : {losses_train_epoch[epoch]:7.5f}, valid loss: {losses_valid_epoch[epoch]:7.5f}")

# Plot training and validation loss
plt.plot(losses_train_epoch, label="trainnig")
plt.plot(losses_valid_epoch, label="validation")
plt.legend(loc="upper right")

# Plot prediction versus target
# Un-normalize the true delta of the AE probability
ae_prob = ds.predictors["ae_prob"].data * ds.predictors["ae_prob"].data_std + ds.predictors["ae_prob"].data_mean
# Un-normalize the delta from the RC model
ae_prob_delta = ds.targets["ae_prob_delta"].data * ds.targets["ae_prob_delta"].data_std + ds.targets["ae_prob_delta"].data_mean
# Calculate the true new value of the AE probabilities
ae_prob_dt_true = ae_prob + ae_prob_delta

# Use the model to get the predicted delta of AE probability
ae_prob_dt_pred = torch.zeros_like(ae_prob_dt_true)
for idx in np.arange(ae_prob_dt_true.shape[0]):
    # Get the predicted delta
    with torch.no_grad():
        ae_prob_delta_pred = model(ds[idx][0]) * ds.targets["ae_prob_delta"].data_std + ds.targets["ae_prob_delta"].data_mean
    # Add predicted delta to AE mode probablity
    ae_prob_dt_pred[idx, :] = ae_prob[idx, :] + ae_prob_delta_pred

# Plot true change in AE mode probability versus 
# predicted change in AE mode probability
# Separate figures for each of the 5 AE mode types
for idx in [0, 1, 2, 3, 4]:
    plt.figure()
    plt.plot(ae_prob_dt_true[:, idx].cpu())
    plt.plot(ae_prob_dt_pred[:, idx].cpu())

    plt.xlim((100, 500))

