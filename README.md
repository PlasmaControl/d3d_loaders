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
t_params = {'tstart' : tstart,
            'tend'   : tend,
            'tsample': tsample
}
shift_targets = {'ae_prob_delta':10.0}

my_ds = D3D_dataset(shotnr, t_params,
                    predictors=["pinj", "neut", "ae_prob"],
                    targets=["ae_prob_delta"],
                    shift_targets=shift_targets,
                    device=device)
```

Here `shotnr` gives the shots to work on, `tstart` and `tend` define a time interval for the
signals, and `tsample` defines the common sampling time for all signals.
`predictors` defines a list of signals that are used to predict the `target`.
`shift_targets` is the amount each time signal should be shifted, except for `ae_prob_delta` 
where it is the interval we are looking into the future to calculate our change in probability. 
They will be stored in the dictionaries `my_ds.predictors` and `my_ds.target`
respectively. Each value of these dicts is a derived class of `signal_1d`.

Currently, the only target is `ae_prob_delta`, calculated using AE probabilities from Aza's RCN model. 
The current predictors are:

| Predictor             | Key     |
|-----------------------|---------|
| AE Mode probabilities | ae_prob |
| Pinj                  | pinj    |
| Neutron Rate          | neut    |
| Injected Power        | ip      |
| ECH                   | ech     |
| q95                   | q95     |
| kappa                 | kappa   |
| Density Profile       | dens    |
| Pressure Profile      | pres    |
| Temperature Profile   | temp    |
| q Profile             | q       |
| Lower Triangularity   | tri_l   |
| Upper Triangularity   | tri_u   |
| Raw ECE Channels      | raw_ece |

At a given index, the output is a list of 2 tensors.
```

In [119]: my_ds[0]
Out[119]: 
(tensor([-0.5374, -0.4721, -0.5583, -0.5301, -0.4714, -1.1228, -1.7159]),
 tensor([-0.0207, -0.0328, -0.0478, -0.0331, -0.0119]))
```

The first tensor has a number of elements fixed by the number of signals being used. 
The order will match the order of the signals in `predictors`. 
Each of these signals is normalized to their separate mean and standard deviation.
The output tensor (2nd tensor) is the change in the Alfven Eigenmode probabilities over the interval given in
`shift_targets` with the `ae_prob_delta` key,
i.e. AE mode probability at t0 + 10ms as calculated using ECE data. This output is also
normalized to zero mean and unit standard deviation.



Iteration over the dataset is done in pytorch-style:
```
for i in torch.utils.data.DataLoader(my_ds):
    # use i
```

An example of how to use this dataloader for predictive modelling is in 'runme.py'

