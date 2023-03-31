D3D Loaders
===========

This package implements Pytorch-style [Iterable Datasets](https://pytorch.org/docs/stable/data.html#)
for D3D data. It contains It contains helper function to fetch data from MDS and store it locally
in HDF5 files. It also contains custom [samplers](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) to easily load batch random sequences from the data. These can be used in conjunction with
[data loaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).



Load Multiple Signals in Single Dataset
=======================================

Import and instantiate the loader like this:
```python
>>> from d3d_loaders.d3d_loaders import D3D_dataset
>>> from d3d_loaders.samplers import  RandomBatchSequenceSampler, collate_fn_random_batch_seq
>>> shotnr = 172337
>>> t_params = {'tstart' : 200.0,
                'tend'   : 1000.0,
                'tsample': 1.0
>>> }
>>> shift_targets = {'ae_prob': 10.0}
>>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


>>> my_ds = D3D_dataset(shotnr, t_params,
>>>                     predictors=["pinj", "tinj", "iptipp", "dstdenp", "doutu", "dssdenest", "ae_prob"],
>>>                     targets=["ae_prob"],
>>>                     shift_targets=shift_targets,
>>>                     datapath="/projects/EKOLEMEN/d3dloader/test",
>>>                     device=device)
```

Here `shotnr` gives the shot from which the signals are used. Together `tstart`, `tend`, and
`tsample` define a common time base for all signals. The signals are resampled in a manner that
conserves causality, that is, no future information is used to generate a single sample.

The dataset defines two groups of signals. `Predictors` are to be used as input to a model and
`targets` are to be used as ground-truth outputs of the model. The dictionary `shift_targets`
defines an offset by which targets are shifted into the future. The name of the signals
refer to either `MDS` nodes or `PTDATA` point names. The names of the signals and their
location in `MDS` or `PTDATA` is defined in the file `downloading.py`. A custom signal,
`ae_prob` refers to output of Aza's Alfven Eigenmode probability model.

Predictor and target signals are stored in the dictionaries `my_ds.predictors` and `my_ds.target` respectively. 

The `datapath` argument defines the directory where the D3D data will be cached
in `HDF5` files. There is one `HDF5` file per shot, which stores the individual signals within
a group. If a signal is not found in that group, it will be fetched automatically and added to
the file. The layout of the files are 

```julia
julia> fid = h5open("/projects/EKOLEMEN/d3dloader/test/172337.h5", "r")
🗂️ HDF5.File: (read-only) /projects/EKOLEMEN/d3dloader/test/172337.h5
├─ 📂 ali
│  ├─ 🏷️ origin
│  ├─ 🔢 xdata
│  │  └─ 🏷️ xunits
│  └─ 🔢 zdata
│     └─ 🏷️ zunits
├─ 📂 doutu
│  ├─ 🏷️ origin
│  ├─ 🔢 xdata
│  │  └─ 🏷️ xunits
│  └─ 🔢 zdata
│     └─ 🏷️ zunits
├─ 📂 dssdenest
│  ├─ 🏷️ origin
│  ├─ 🔢 xdata
│  └─ 🔢 zdata
├─ 📂 dstdenp
│  ├─ 🏷️ origin
│  ├─ 🔢 xdata
│  └─ 🔢 zdata
...
```

Each group is named after the shortname signal and contains `xdata`, `zdata` nodes returned from `gadata.py`.
Additionally, the origin of the data is stored in the `origin` attribute, and the units of the signals
are stored in the `xunits` and `zunits` attributes, when available.


Additionally, a `device` can be passed. If set to be the GPU, the `predictor` and 
`target` tensors will be stored on the GPU. This avoids having to load batches into GPU memory in the
training data loop.



Iterating over signal sequences in a single shot
================================================
To fetch data sequences from a single shot, we can instantiate a `DataLoader`. This package
implements a `RandomBatchSequenceSampler` that fetches linear sequences from both, `predictor`
and `target` tensors of the dataset. `RandomBatchSequenceSampler` fetches a batch of sequences,
all of the same length and starting at the same random point. The sample below shows how 
`D3D_dataset` and `RandomBatchSequenceSampler` work together to allow iteration over the sequences:

```python
>>> len(ds)
    800
>>> batch_size = 32
>>> seq_length = 512
>>> sampler = BatchedSampler(len(ds), seq_length=seq_length, batch_size=batch_size, shuffle=True)
>>> loader_train = torch.utils.data.DataLoader(my_ds, num_workers=0, 
                                               batch_sampler=sampler,
                                               collate_fn = collate_fn_batched)

>>> for x_b, y_b in loader_train:
        print(f"x_b.shape={x_b.shape}, y_b.shape={y_b.shape}")

x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
x_b.shape=torch.Size([32, 513, 11]), y_b.shape=torch.Size([32, 513, 5])
```

In each iteration, `loader_train` returns a tuple of tensors. The first tensor contains
`32=batch_size` sequences of length `seq_length+1`. Each sequence has `11` features, which
is the sum of the feature dimension of each `predictor` signal. The second tensor has the
same batch size and sequence length, but the feature dimension is `5`, corresponding to
the probability of each Alfven Eigenmode.

The length of the sequences is `seq_length+1=513`. Note also, that the `predictor` signals are
shifted `10ms` into the future. That is the samples at `y_b[:, -1, :]`  are `11ms` ahead
of the samples in `x_b[:, -2, :]`. 

In the example above, `collate_fn_batched` reshapes the returned data
from `d3d_dataset.__getidx__` into a tuple of 2 tensors. The call semantics are explained in the
code example [here](https://pytorch.org/docs/stable/data.html#disable-automatic-batching).
Note that `RandomBatchSequenceSampler` takes care of batching. Automatic batching in the pytorch
`DataLoader` needs to be disabled.


The `BatchedSampler` can either sample sequences linearly, starting at 0, or shuffle
the starting points of the sequences.

For `shuffle=False`, `BatchedSampler` generates sequences where the starting index shifts by 1 for
each sequence in a batch. Using a `BatchedSampler` in the example above would have
the first sequence start at index 0 and extending to 512. The second sequence
would start at index 1 and extend to 513. And so on.

This is useful for inference, where we may want to predict a target sequentially over the 
entire shot. We input samples of all predictors at time index 0...511 into the model and get a 
prediction for the target at time index 512. Then we input the predictors at time index
1..512 to predict the target at time index 513. And so in. Thus, by iterating over a
DataLoader which uses a `BatchedSampler`, we can easily get a reconstruction of the
predicted target for the entire shot.

Setting `shuffle=True`, `BatchedSampler` generates sequences that start at a random point
in the dataset. Looking at the code above, the first of the 32 sequences in the batch
may start at index 18 (and extend to index 18+513=631) of the entire shot.
The second sequence may start at index 788 and extend to index 788+513. 
So every starting index is at random. This is useful for training.


There are also the versions of the sampler `BatchedSampler_dist`. These should
be used for distributed training. They effectively distribute all available samples
in the dataset across ranks.



Multi-shot datasets
===================

THIS IS A BIT OUTDATED. THE INTERFACE TO THE MULTISHOT DATASET HAS CHANGED.
SEE [https://github.com/PPPLDeepLearning/frnn_examples](https://github.com/PPPLDeepLearning/frnn_examples) FOR PROPER USAGE.

The class `Multishot_dataset` defines a dataset that spans multiple shots. It can be instantiated in
a very similar manner:

```python

>>> from d3d_loaders.d3d_loaders import Multishot_dataset
>>> from d3d_loaders.samplers import BatchedSampler_multi, collate_fn_batched
>>> shot_list_train = [172337, 172339] 
>>> tstart = 110.0 # Time of first sample for upper triangularity is 100.0
>>> tend = 2000.0
>>> t_params = {"tstart": tstart, "tend": tend, "tsample": 1.0}
>>> t_shift = 10.0
>>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>>> seq_length = 512
>>> batch_size = 4
>>> pred_list = ["pinj",  # Injected power
>>>              "tinj", # Injected torque
>>>              "iptipp",  # Target current
>>>              "dstdenp", # Target density
>>>              "doutu", # Top triangularity
>>>              "dssdenest", # Line-averaged density
>>>              "ae_prob"] # AE mode probability
>>> targ_list = ["ae_prob"]

>>> ds_train = Multishot_dataset(shot_list_train, t_params, pred_list, targ_list,
>>>                              {"ae_prob": t_shift}, 
>>>                              datapath="/projects/EKOLEMEN/d3dloader/test", device)
```

To iterate over a multi-shot database, we can again instantiate a DataLoader using the custom
`RandomBatchSequenceSampler_multids` sampler as well as a custom `collate_fn`:

```python
>>> for xb, yb in loader_train_b:
>>>     print(xb.shape, yb.shape)
>>>    break

torch.Size([513, 4, 11]) torch.Size([513, 4, 5])
```

Again, the tensor tuple `xb` and `yb` contain a batch of random sequences. But the sequences are picked
at random from the list of shots used to construct the `Multishot_dataset`. All sequences are constructed
from data of only a single shot.


Downloading signals
===================
MDS and PTDATA can be downloaded with the script `downloading.py`. This script fetches data for
a given shot from D3D's MDSplus server and stores in in a HDF5 file.

The current predictors are:

| Predictor             | Key             |
|-----------------------|-----------------|
| AE Mode probabilities | ae_prob         |
| Pinj                  | pinj            |
| Neutron Rate          | neutronsrate    |
| Injected Power        | ip              |
| ECH                   | ech             |
| q95                   | q95             |
| kappa                 | kappa           |
| Density Profile       | dens            |
| Pressure Profile      | pres            |
| Temperature Profile   | temp            |
| q Profile             | q               |
| Lower Triangularity   | doutl           |
| Upper Triangularity   | doutl           |
| Raw ECE Channels      | raw_ece         |
| Raw CO2 dp Channels   | raw_co2_dp      |
| Raw CO2 pl Channels   | raw_co2_pl      |
| Raw MPI Channels      | raw_mpi         |
| Raw BES Channels      | raw_bes         |
| Bill's AE Labels (+-250ms window) | uci_label |

NOTE: the shape signal of kappa and shape profiles of upper and lower triangularity don't have data even though
the keys exist in the hdf5 files. You should get a helpful error telling you this, however you may also get
an error telling you of an array splicing error. Also when these signals do exist, the first signal is very 
late, so make sure `tstart` is sufficiently large. 

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

Currently, the only target is `ae_prob_delta`, calculated using AE probabilities from Aza's RCN model. 
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

Load Full Resolution Single Signal
==================================
By setting `tsample=-1`, the full resolution will return with the `tstart` and `tend` 
found closest to the true measurement values (forwards or backwards in time). This will
not load as a dataset since you cannot temporally align the full resolution signals. 
An example for loading  neutron rate into a numpy array is below

```python
from d3d_loaders.signal0d import signal_neut
shotnr = 169113
t_params = {'tstart' : tstart,
            'tend'   : tend,
            'tsample': -1
}

sig_neut = signal_neut(shotnr, t_params)

data = sig_neut.data.numpy()
```



Full training examples
======================

Training examples can be found in [this repository](https://github.com/PPPLDeepLearning/frnn_examples).
The dataset these examples operate on can be found [here](https://github.com/PPPLDeepLearning/dataset_D3D_100).




The folder `examples` contains several notebooks that illustrate how to train predictive
models using the infrastructure provided in this package. In particular, the notebook
`AE_pred_LSTMtest_1723xx.ipynb` shows how to train a LSTM-based model for predictive
tasks. And the nodebook `AE_pred_transformer_154xxx.ipynb` shows how to train a transformer-based
model for predictive tasks. Both notebooks rely on the dataset and dataloader infrastructure
implemented in this package. 

The datasets used by the notebooks are located on Princeton's gpfs, accessible on
`stellar` and `traverse`: `stellar:/projects/EKOLEMEN/AE_datasets`. They have been compiled
using the [d3d_signals](https://github.com/PlasmaControl/d3d_signals) package. Definitions of
the datasets in `yaml` format is [here](https://github.com/PlasmaControl/AE_datasets). 
There is an additional `README` with notes on issues that arose when compiling the dataset. 


