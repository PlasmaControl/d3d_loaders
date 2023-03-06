#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch

class standardizer_mean_std():
    """Standardizes data to mean/std.

    """
    def __init__(self, mean, std):
        """Set mean and std. So far only scalar tested.

        mean : float
        std : float
        """
        self.mean = mean
        self.std = std

    def __call__(self, signal):
        return (signal - self.mean) / (self.std + 1e-10) 


    def __repr__(self):
        return f"standardizer_mean_std, mean={self.mean}, std={self.std}"



# end of file standardizers.py