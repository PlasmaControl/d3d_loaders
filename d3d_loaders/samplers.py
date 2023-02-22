# -*- coding: UTF-8 -*-

"""Sampler classes that aid in training recurrent neural networks on D3D time series.

"""

import torch
from torch.utils.data import SequentialSampler, BatchSampler, RandomSampler, Sampler
from typing import Sequence, Iterator
import logging

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
        self.indices = range(num_elements)
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        print(f"__init__() indices={self.indices}, seq_length={self.seq_length}, batch_size={self.batch_size}")

        if self.batch_size >= num_elements:
            raise ValueError("Batch size must be smaller than the size of the dataset. ")

    def __iter__(self) -> Iterator[int]:
        """Returns a batch of fixed length sequences, starting at random."""
        # Define random starting indices of the sequence
        idx_permuted = torch.randperm(len(self.indices) - self.seq_length)
        print(f"idx_permuted = {idx_permuted}, size = {idx_permuted.shape}")
        print(f"batch_size = {self.batch_size}")
        # Number of batches to draw. Round up.
        num_batches = (len(self.indices) - self.seq_length) // self.batch_size
        for ix_b in range(0, num_batches):
            # draw (batch_size) starting indices
            ix_start = idx_permuted[(ix_b * self.batch_size):((ix_b + 1) * self.batch_size)]
            # Return a list of tensors, so that the entire tensor is passed to dataset.__idx__:
            # See call for map-style datasets in code example here: https://pytorch.org/docs/stable/data.html#automatic-batching-default
            yield [torch.cat([torch.arange(i, i + self.seq_length + 1) for i in ix])]

class SequentialSequenceSampler(Sampler[int]):
    r"""Samples sequences randomly, without replacement.
    
    Given a dataset of length N, this sampler will generate (N-seq_length-1)
    sequences of length (seq_length + 1). The starting index of these
    (N - seq_length - 1) sequences is in order.
    
    # Generate sequences of length 3, as to cover [0, 1, 2, 3, 4, 5]
    >>> rs = SequentialSequenceSampler([0, 1, 2, 3, 4, 5], 2)
    >>> for item in rs:
            print(item)
    0
    1
    2
    1
    2
    3
    2
    3
    4
    3
    4
    5


    In the output above, the first sequence of length (2 + 1) starts at 0: [0, 1, 2].
    The second sequence of length (2 + 1) starts at 1: [1, 2, 3].
    The third sequence of length (2 + 1) starts at 2: [2, 3, 4].
    And the fourth sequence of length (2 + 1) starts at 3: [3, 4, 5].
    
    Thus, we have exhausted the possibility of many-to-one mappings of the kind:
    
    f(x_{i}, x_{i+1}, ..., x_{i + seq_length - 1}) -> y_{i + seq_length}
    
    Args:
        seq_length: Length of the desired sequence
    """
    indices: Sequence[int]
        
    def __init__(self, indices: Sequence[int], seq_length: int) -> None:
        self.indices = indices
        self.seq_length = seq_length
        
    def __iter__(self) -> Iterator[int]:
        """Returns fixed-length sequences, starting at random."""
        # Iterate over all possible start indices
        # Substract seq_length + 1, so that the last element can be included as a prediciton target
        for start in range((len(self.indices) - self.seq_length)):
            # The yield seq_length successive indices
            for seq_idx in self.indices[start:start+self.seq_length + 1]:
                yield seq_idx
            
    def __len__(self) -> int:
        return len(self.indices)



def collate_fn_batched(ll, seq_length):


def collate_fn_randseq(ll, seq_length):
    """Works in combination with RandomSequenceSampler.
    
    X list needs to skip every (seq_length+1) sample
    Y list needs to take only every (seq_length+1) sample
    
    The input samples is a list of the dataset ds, evaluated with indices
    idx_list = [i, i+1, i+2, ..., i + seq_length, j, j+1, j+2, ..., j + seq_length, ..., k, k+1, ... k + seq_length].
    Here i, j, ..., k, are various indices that define the start of a sequence with length seq_length.
    
    That is, ll = [dataset[idx] for idx in idx_list]
    
    This collate_fn is specific to the D3D_dataset, where ds[idx] = (X[idx], Y[idx]).
    In particular, it re-orders ll = [(X[idx], Y[idx]) for idx in idx_list] 
    into
    
    X[L, N, H] 
    Y[1, N, H] 
    where L: sequence length, N: number of batches, H: number of features
    
    Y is assumed to have sequence length=1.
    
    The in
    X[i], X[i+1], ... X[i + seq_length - 1] -> Y[i + seq_length]
    X[j], X[j+1], ... X[j + seq_length - 1] -> Y[j + seq_length]
    ...
    X[k], X[k+1], ... X[k + seq_length - 1] -> Y[k + seq_length]
    
    Args:
        ll: Sequence[int] Sequence of index groups, 
    
    """
    # List skipping code inspired by: https://stackoverflow.com/questions/40929560/skip-every-nth-index-of-numpy-array
    # List of indices we take for X:
    all_idx = [i for i in range(len(ll))]
    
    # Define the indices for the predictor tensors X.
    x_list_idx = [i for i in all_idx]
    # This expression will skip every (seq_length+1)-th element in all_idx
    del x_list_idx[(seq_length + 1) - 1::(seq_length + 1)]
    # Starting at seq_length, take every (seq_length+1)-th element in all_idx
    y_list_idx = all_idx[seq_length::(seq_length + 1)]
           
    # Stack feature tensor
    x = torch.cat([ll[idx][0].unsqueeze(0) for idx in x_list_idx], dim=0)
    LN, H = x.shape        # First dimension is product of sequence_length * batch_size
    N = LN // seq_length
    # Reshape X to (L, N, H) where L: sequence length, N: batch size and H: input_size
    x = x.reshape(N, seq_length, H).transpose(0,1)
    
    # Stack target tensor
    y = torch.cat([ll[idx][1].unsqueeze(0) for idx in y_list_idx], dim=0).unsqueeze(0)
   
    return x, y

# end of file samplers.py