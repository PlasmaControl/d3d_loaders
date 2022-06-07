#!/usr/bin/env python
import torch
from d3d_loaders.d3d_loaders import D3D_dataset

# Instantiate the D3D Loader object
my_ds = D3D_dataset()

# Print all available shots and time
print(my_ds.ece_label_df[["shot", "time"]])

# Test iteration
num_iter = 0
for i in torch.utils.data.DataLoader(my_ds, num_workers=1):
    print(num_iter) 
    num_iter += 1

print(f"{num_iter} iterations")
print(type(i[0][0][0]), i[0][0][0].dtype)
