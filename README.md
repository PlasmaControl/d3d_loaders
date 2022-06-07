D3D Loaders
===========

Implements data loader for D3D data. The loader itself is targeted to load data from the HDF5
files compiled in /projects/EKOLEMEN/aza_lenny_data1. Its structure is copied off of Pytorch's
[Iterable Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

Examples
========

Import and instantiate the loader like this:
```python
from d3d_loaders.d3d_loaders import D3D_dataset
my_ds = D3D_dataset()
```

Iteration over the dataset is done in pytorch-style:
```
for i in torch.utils.data.DataLoader(my_ds):
    # use i
```
