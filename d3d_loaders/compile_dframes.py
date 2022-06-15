#!/usr/bin/env python
from os.path import join
import pandas

from d3d_loaders.rcn_functions import preprocess_label

def process_df_bill(datapath):
    """Processes Bill's list of AE labels.

    Built in to this dataset is a set of (shot, t0). These are defined by
    the intersection of the shots and time points defined in `uci_ae_labels.txt`
    and those shots for which neutronrate data is available.
    """
    ece_label_df = preprocess_label(join(datapath,  "uci_ae_labels.txt"))

    # Load shots for which neutron rates are available. This provides the second set of (shots,)
    shots_neutrons = []
    with open(join(datapath, "shots_neutrons.txt"), "r") as f:
        for line in f.readlines():
            shots_neutrons.append(int(line))    

    # Now form the intersection of both sets of shots.
    # The result is a list of shots which we have ece labels and neutron rates
    shot_list = list(set(shots_neutrons).intersection(set(ece_label_df.shot.tolist())))
    shot_list.sort()
    # Remove all rows for which there is no neutron data
    ece_label_df = ece_label_df[ece_label_df["shot"].isin(shot_list)]

    # Shot 171997 is bad. Drop it
    ece_label_df = ece_label_df[ece_label_df.shot != 171997]

    return ece_label_df
