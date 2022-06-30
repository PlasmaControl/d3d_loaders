D3D Loaders
===========

Implements data loader for D3D data. The loader itself is targeted to load data from the HDF5
files compiled in `/projects/EKOLEMEN/aza_lenny_data1`. 
Its structure is copied off of Pytorch's
[Iterable Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

Examples
========

Import and instantiate the loader like this:
```python
from d3d_loaders.d3d_loaders import D3D_dataset
my_ds = D3D_dataset(shotnr, tstart, tend, tsample,
                    predictors=["pinj", "neut", "ae_prob"],
                    targets=["ae_prob_delta"],
                    shift_target=10.0,
                    device=device)
```

Here `shotnr` gives a shot to work on, `tstart` and `tend` define a time interval for the
signals, and `tsample` defines the common sampling time for all signals.
`predictors` defines a list of signals that are used to predict the `target`.
They will be stored in the dictionaries `my_ds.predictors` and `my_ds.target`
respectively. Each value of these dicts is a derived class of `signal_1d`.

Currently the loader uses the Alfven Eigenmode probability, as output by Aza's RCN model,
summed neutral power, and neutron rate as the data sources.

At a given index, the output is a list of 2 tensors.
```

In [119]: my_ds[0]
Out[119]: 
(tensor([-0.5374, -0.4721, -0.5583, -0.5301, -0.4714, -1.1228, -1.7159]),
 tensor([-0.0207, -0.0328, -0.0478, -0.0331, -0.0119]))
```

The first tensor has 7 elements. The first 5 are the probabilities for a given Alfven
Eigenmode, element 6 gives the neutron rate, and element 7 encodes summed pinj.
Each of these three groups is normalized to their separate mean and standard deviation.
The output is the change in the Alfven Eigenmode probabilities over a 50 microsecond interval,
i.e. AE mode probability at t0 + 50mus as calculated using ECE data. This output is also
normalized to zero mean and unit standard deviation.



Iteration over the dataset is done in pytorch-style:
```
for i in torch.utils.data.DataLoader(my_ds):
    # use i
```

An example of how to use this dataloader for predictive modelling is in 'runme.py'

