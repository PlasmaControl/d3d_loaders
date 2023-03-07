# -*- coding: UTF-8 -*-

"""Sampler classes that aid in training recurrent neural networks on D3D time series.

"""

import torch
from torch.utils.data import Sampler
from typing import Sequence, Iterator
import random
import logging



class SequentialSampler(Sampler):
    r"""Samples sequences in-order
    
    Given a dataset of length N, this sampler will generate (N-seq_length-1)
    sequences of length (seq_length + 1). The starting index of these
    (N - seq_length - 1) sequences is in order.
    
    # Generate sequences of length 3, as to cover [0, 1, 2, 3, 4, 5]
    >>> my_sampler = SequentialSampler(6, 2)
    >>> for s in my_sampler:
    >>>     print(s)
    [range(0, 3)]
    [range(1, 4)]
    [range(2, 5)]


    In the output above, the first sequence of length (2 + 1) starts at 0: range(0,3) = [0, 1, 2].
    The second sequence of length (2 + 1) starts at 1: range(1, 4) = [1, 2, 3].
    
    Thus, we have exhausted the possibility of many-to-one mappings of the kind:
    
    f(x_{i}, x_{i+1}, ..., x_{i + seq_length - 1}) -> y_{i + seq_length}
    
    Args:
        ds_length: Length of the dataset.
        seq_length: Length of the sequence
    """ 
    def __init__(self, ds_length, seq_length):
        self.ds_length = ds_length # Length of the dataset
        self.seq_length = seq_length # Length of the sequences to sample
        
    def __iter__(self):
        """Returns fixed-length, ordered sequences that cover the dataset."""
        for start in range(self.ds_length - self.seq_length - 1):
            yield [range(start, start + self.seq_length + 1)]

class collate_fn_seq():
    r"""Functor to be used in DataLoader with SequentialSampler.

    ```
    >>> loader_train_seq = torch.utils.data.DataLoader(ds, num_workers=0, 
    >>>                                          batch_sampler=SequentialSampler(len(ds), seq_length=seq_length),
    >>>                                          collate_fn = collate_fn_seq())
    >>> for x, y in loader_train_seq:
    >>> print(x.shape, y.shape)
        torch.Size([513, 11]) torch.Size([513, 5])
        torch.Size([513, 11]) torch.Size([513, 5])
        ...
    ```

    """
    def __init__(self):
        None
        
    def __call__(self, samples):
        return samples[0]


# class SequentialSampler(Sampler[int]):
#     r"""Samples sequences in-order
    
#     Given a dataset of length N, this sampler will generate (N-seq_length-1)
#     sequences of length (seq_length + 1). The starting index of these
#     (N - seq_length - 1) sequences is in order.
    
#     # Generate sequences of length 3, as to cover [0, 1, 2, 3, 4, 5]
#     >>> my_sampler = SequentialSampler(6, 2)
#     >>> for s in my_sampler:
#     >>>     print(s)
#     [range(0, 3)]
#     [range(1, 4)]
#     [range(2, 5)]


#     In the output above, the first sequence of length (2 + 1) starts at 0: range(0,3) = [0, 1, 2].
#     The second sequence of length (2 + 1) starts at 1: range(1, 4) = [1, 2, 3].
    
#     Thus, we have exhausted the possibility of many-to-one mappings of the kind:
    
#     f(x_{i}, x_{i+1}, ..., x_{i + seq_length - 1}) -> y_{i + seq_length}
    
#     Args:
#         ds_length: Length of the dataset.
#         seq_length: Length of the sequence
#     """ 
#     def __init__(self, ds_length, seq_length):
#         self.ds_length = ds_length # Length of the dataset
#         self.seq_length = seq_length # Length of the sequences to sample
        
#     def __iter__(self):
#         """Returns fixed-length, ordered sequences that cover the dataset."""
#         for start in range(self.ds_length - self.seq_length - 1):
#             yield [range(start, start + self.seq_length + 1)]


class SequentialSamplerBatched(Sampler):
    r"""Sample linear sequences in order, allows for batching.

    Similar to SequentialSampler, but returns a batch of sequences in each iteration.
    """
    def __init__(self, ds_length, seq_length, batch_size):
        self.ds_length = ds_length      # Length of the dataset
        self.seq_length = seq_length    # Length of the sequences to sample
        self.batch_size = batch_size    # Batch size
        
    def __iter__(self):
        """Returns fixed-length, ordered sequences that cover the dataset."""
        for start in range(0, self.ds_length - self.seq_length - 1, self.batch_size):
            yield [range(start + bs, start + bs + self.seq_length + 1) for bs in range(self.batch_size) if start + bs + self.seq_length + 1 <= self.ds_length]


class collate_fn_seq_batched():
    """Stacks list of sequences into a single tensor.

    ```python
    >>> batch_size = 32
    >>> seq_length = 513
    >>> loader_train_seq_batched = torch.utils.data.DataLoader(ds, num_workers=0, 
    >>>                                                        batch_sampler=SequentialSamplerBatched(len(ds), seq_length, batch_size),
    >>> for xb, yb in loader_train_seq_batched:
    >>>     print(xb.shape, yb.shape)
        torch.Size([513, 32, 11]) torch.Size([513, 32, 5])
        torch.Size([513, 32, 11]) torch.Size([513, 32, 5])
        ...
    """
    def __init__(self):
        None
        
    def __call__(self, samples):
        x_stacked = torch.cat([s[0][None, :, :] for s in samples], dim=0)
        y_stacked = torch.cat([s[1][None, :, :] for s in samples], dim=0)
        return x_stacked, y_stacked


class SequentialSamplerBatched_multi(Sampler):
    r"""Sample batched, linear sequences from multishot dataset."""
    def __init__(self, num_shots:int, num_elements: int, seq_length: int, batch_size: int) -> None:
        self.num_shots = num_shots
        self.num_elements = num_elements
        self.seq_length = seq_length
        self.batch_size = batch_size

    def __iter__(self):
        """Return a batch of linear sequences.
        
        * Always exhaust one dataset, even if it means that the batch will be smaller than 
          requested batch_size, before continuing on the next shot.
        """
        for s in range(0, self.num_shots):
            for start in range(0, self.num_elements - self.seq_length - 1, self.batch_size):
                yield [(s, range(start + b, start + b + self.seq_length + 1)) for b in range(self.batch_size) if start + b + self.seq_length + 1 <= self.num_elements]
        




class RandomSequenceSampler(Sampler[int]):
    r"""Samples sequences randomly, without replacement.
    
    Given a dataset of length N, this sampler will generate (N-seq_length-1)
    sequences of length (seq_length + 1). The starting index of these
    (N - seq_length - 1) sequences is random.
    
    # Generate sequences of length 3, as to cover [0, 1, 2, 3, 4, 5]
    >> torch.manual_seed(1337)
    >>> rs = RandomSequenceSampler(5, 2)
    >>> for item in rs:
            print(item)

    range(1, 4)
    range(2, 5)
    range(0, 3)

    In the output above, the first sequence of length (2 + 1) covers items 1, 2, 3 = range(1, 4)
    The second sequence of length (2 + 1) covers items 2, 3, 4 = range(2, 5)
    The third sequence of length (2 + 1) covers items 0, 1, 2 = range(0, 3)
    
    Thus, we have exhausted the possibility of many-to-one mappings of the kind:
    
    f(x_{i}, x_{i+1}, ..., x_{i + seq_length - 1}) -> y_{i + seq_length}
    
    
    Args:
        num_elements: Length of the dataset
        seq_length: Length of the desired sequence
    """
    indices: Sequence[int]
        
    def __init__(self, num_elements: int, seq_length: int) -> None:
        self.indices = range(num_elements)
        self.seq_length = seq_length
        
    def __iter__(self) -> Iterator[int]:
        """Returns fixed-length sequences, starting at random."""
        # Iterate over all possible start indices
        # Substract seq_length + 1, so that the last element can be included as a prediciton target
        for start in torch.randperm(len(self.indices) - self.seq_length):
            #print("start_idx = ", start)
            # The yield seq_length successive indices
            yield self.indices[start:start+self.seq_length + 1]
            
    def __len__(self) -> int:
        return len(self.indices)


class RandomBatchSequenceSampler(Sampler[int]):
    r"""Randomly samples batched sequences without replacement.

    """

    indices: Sequence[int]

    def __init__(self, num_elements: int, seq_length: int, batch_size: int) -> None:
        self.num_elements = num_elements    # Number of elements in the sequence we want to sample
        self.seq_length = seq_length        # Length of the sequence we want to sample
        self.batch_size = batch_size        # Batch size
        
        if self.batch_size >= num_elements:
            raise ValueError("Batch size must be smaller than the size of the dataset. ")

    def __iter__(self) -> Iterator[int]:
        """Returns a batch of fixed length sequences, starting at random."""
        # Define random starting indices of the sequence. Remember sequence length = self.seq_length + 1.
        idx_permuted = list(range(self.num_elements - self.seq_length - 1))
        random.shuffle(idx_permuted)
        # Number of batches to draw. Round down
        num_batches = (self.num_elements - self.seq_length - 1) // self.batch_size
        for ix_b in range(0, num_batches):
            # draw (batch_size) starting indices
            ix_start = idx_permuted[(ix_b * self.batch_size):((ix_b + 1) * self.batch_size)]
            # Return a list of tensors, so that the entire tensor is passed to dataset.__idx__:
            # See call for map-style datasets in code example here: https://pytorch.org/docs/stable/data.html#automatic-batching-default
            yield [torch.cat([torch.arange(i, i + self.seq_length + 1) for i in ix_start])]


class collate_fn_random_batch_seq():
    """Reshape output returned by RandomBatchSequenceSampler.

    Output has shape (N, L, H), corresponding to batch_first=True in
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def __call__(self, x):
        x = x[0]
        # Unpack the stacked indices from RandomBatchSequenceSampler.__iter__
        x_stacked = x[0].reshape(self.batch_size, -1, x[0].shape[-1])
        y_stacked = x[1].reshape(self.batch_size, -1, x[1].shape[-1])
        return x_stacked, y_stacked


class RandomBatchSequenceSampler_multishot():
    r"""Randomly samples batched sequences from multi-shot dataset without replacement.
    
    Works similar to RandomBatchSequenceSampler, but spreads sampling out over multiple datasets.
    As of now (2023-02), all datasets have to have the same length.
    
    Args:
        num_elements (List[Int]): Elements per dataset.
        seq_length: Length of sequences to sample
        batch_size: Number of sequences to return per iteration.

    """

    #indices: Sequence[int]

    def __init__(self, num_shots:int, num_elements: int, seq_length: int, batch_size: int) -> None:
        self.num_shots = num_shots
        self.num_elements = num_elements
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        if self.batch_size >= min(num_elements):
            raise ValueError("Batch size must be smaller than the size of the dataset. ")

    def __iter__(self): # -> Iterator[int]:
        """Returns a batch of fixed length sequences, starting at random."""
        # Randomly shuffle starting indices for each shot
        idx_permuted = [(s, i) for i in range(self.num_elements - self.seq_length) for s in range(self.num_shots)]
        random.shuffle(idx_permuted)

        # Number of batches to draw. Round up.
        num_batches = self.num_shots * (self.num_elements - self.seq_length) // self.batch_size
        for ix_b in range(0, num_batches):
            # Select starting points for sequences
            selected = idx_permuted[(ix_b * self.batch_size):((ix_b + 1) * self.batch_size)]
            # Remember to return a list. PyTorch dataloader passes each item in the
            # returned list to dataset.__getidx__. If we only return a single list,
            # each scalar index in that list would be passed to __getidx__.
            # If we return a list of lists, that inner list will be passed to __getidx__.
            # Then this list will be used for indexing.
            # Long story short: pass list of lists, not a single list.
            yield [(s[0], range(s[1], s[1] + self.seq_length + 1)) for s in selected]


class collate_fn_random_batch_seq_multi():
    r"""Stacks samples from RandomBatchSequenceSampler_multishot into single tensors.
    Output should have shape (L, N, H), batch_first=False:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def __call__(self, samples):
        x_stacked = torch.cat([s[0][:, None, :] for s in samples], dim=1)
        y_stacked = torch.cat([s[1][:, None, :] for s in samples], dim=1)
        return x_stacked, y_stacked

# end of file samplers.py