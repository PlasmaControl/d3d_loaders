# -*- coding: utf-8 -*-

# run as
# python -m unittest tests/test_samplers.py

from d3d_loaders.samplers import SequentialSequenceSampler

import unittest
class test_sampler_sequential(unittest.TestCase):
    def test_small(self):
        item_list = []
        ss = SequentialSequenceSampler([0, 1, 2, 3, 4, 5], 2)
        item_list = [item for item in ss]
        assert(item_list == [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5])

        



# end of file test_samplers.py