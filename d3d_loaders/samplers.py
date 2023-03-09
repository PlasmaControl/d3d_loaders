# -*- coding: UTF-8 -*-

"""Sampler classes that aid in training recurrent neural networks on D3D time series.

"""

import torch
from torch.utils.data import Sampler
from typing import Sequence, Iterator
import random
import logging




class BatchedSampler(Sampler):
    r"""Sample linear sequences, allows for batching and shuffling.

    Similar to SequentialSampler, but returns a batch of sequences in each iteration.
    """
    def __init__(self, num_elements, seq_length, batch_size, num_replicas=None, rank=None, shuffle=False, seed=0):
        self.num_elements = num_elements  # Length of the dataset
        self.seq_length = seq_length      # Length of the sequences to sample
        self.batch_size = batch_size      # Batch size
        self.num_replicas = num_replicas  # MPI_COMM_WORLD size
        self.rank = rank                  # rank of current worker
        self.shuffle = shuffle            # Shuffle the start of the sequences?
        self.seed = seed                  # Seed for shuffling
        self.epoch = 0                    # Increase this after each epoch to get different shuffling in next iteration

        # Code below copied from https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        # Default initialization matches rank/world size.
        # keep None if we don't use distributed.
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval"
                f" [0, {num_replicas - 1}]")
    
    def set_epoch(self, epoch):
        """Update epoch to adjust random seed."""
        self.sepoch = epoch
        
    def __iter__(self):
        """Returns fixed-length, ordered sequences that cover the dataset."""
        idx_permuted = [(ix) for ix in range(self.num_elements - self.seq_length )]
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(idx_permuted)

        # sub-sample per rank. Each rank starts at a different offset, skipping num_replicas samples
        idx_permuted = idx_permuted[self.rank:self.num_elements:self.num_replicas]
                      
        # Slicing the list like this takes care of partial batches
        for start in range(0, len(idx_permuted), self.batch_size):
            yield [range(ix, ix + self.seq_length + 1) for ix in idx_permuted[start:start+self.batch_size]]


class BatchedSampler_multi():
    r"""Randomly samples batched sequences from multi-shot dataset without replacement.
    
    Works similar to RandomBatchedSampler, but spreads sampling out over multiple datasets.
    
    Args:
        num_elements (List[Int]): Elements per dataset.
        seq_length (Int) : Length of sequences to sample
        batch_size (Int) : Number of sequences to return per iteration.
        num_replicas (Int) : Number of processes participating in distributed training
        rank (Int): Rank of the current processes. By default, retrieved from distributed training group
    """
    def __init__(self, num_elements, seq_length, batch_size, num_replicas=None, rank=None, shuffle=False, seed=0):
        self.num_elements = num_elements
        self.num_shots = len(num_elements)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if self.batch_size >= sum(num_elements):
                raise ValueError("Batch size must be smaller than the size of the dataset. ")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval"
                f" [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        
   
    def set_epoch(self, epoch):
        """Sets epoch for this sampler.
        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        
        Args:
            epoch (int) : Epoch number
        """
        self.epoch = epoch

    def __iter__(self): # -> Iterator[int]:
        """Returns a batch of fixed length sequences, starting at random."""
        # Randomly shuffle starting indices for each shot
        idx_permuted = [(s, i)  for s in range(self.num_shots) for i in range(self.num_elements[s] - self.seq_length)]
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(idx_permuted)

        # Sub-sample for replicas.
        ll = len(idx_permuted)
        idx_permuted = idx_permuted[self.rank:ll:self.num_replicas]
        # After rank sub-sampling, do batching on the sub-sampled elements.
        
        full_batches = len(idx_permuted) // self.batch_size # Number of batches we can fill with the specified batch size
        # Check if the last batch is full or partial
        # We iterate up to num_batches. If in the loop the batch_counter == full_batches, we will have a partial patch
        if len(idx_permuted) != full_batches * self.batch_size:
            remaining_samples = len(idx_permuted) - full_batches * self.batch_size
            num_batches = full_batches + 1
        else: 
            num_batches = full_batches
     
        for ix_b in range(0, num_batches):
            # If ix_x is full_batches (remember 0-based indexing and num_batches is excludede in range)
            # we have need to fill a partial batch with the remaining samples.
            if ix_b == full_batches:  
                selected = idx_permuted[-remaining_samples:]
            else:
                # Fill a full batch
                # Select starting points for sequences
                selected = idx_permuted[(ix_b * self.batch_size):((ix_b + 1) * self.batch_size)]
            # Remember to return a list. PyTorch dataloader passes each item in the
            # returned list to dataset.__getidx__. If we only return a single list,
            # each scalar index in that list would be passed to __getidx__.
            # If we return a list of lists, that inner list will be passed to __getidx__.
            # Then this list will be used for indexing.
            # Long story short: pass list of lists, not a single list.
            yield [(s[0], range(s[1], s[1] + self.seq_length + 1)) for s in selected]



class collate_fn_batched():
    """Stacks list of sequences into a single tensor.

    Can be used for both, BatchedSampler and BatchedSampler_multi.
    Output have shape (L, N, H), batch_first=False:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

    ```python
    >>> batch_size = 32
    >>> seq_length = 512
    >>> loader = torch.utils.data.DataLoader(ds,
    >>>                                      batch_sampler=BatchedSampler(len(ds), seq_length, batch_size),
    >>>                                      collate_fn=collate_fn_batched())
    >>> for xb, yb in loader:
    >>>     print(xb.shape, yb.shape)
        torch.Size([513, 32, 11]) torch.Size([513, 32, 5])
        torch.Size([513, 32, 11]) torch.Size([513, 32, 5])
        ...
    """
    def __init__(self):
        None
        
    def __call__(self, samples):
        x_stacked = torch.cat([s[0][:, None, :] for s in samples], dim=1)
        y_stacked = torch.cat([s[1][:, None, :] for s in samples], dim=1)
        return x_stacked, y_stacked

# end of file samplers.py