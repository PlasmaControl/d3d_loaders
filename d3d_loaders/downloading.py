# -*- coding: utf-8 -*-

"""Example script to download MDSplus / PTDATA 

This script downloads a list of signals for a given shot.
The data is stored in HDF5 format with a layout compatible
with the data loading logic of the d3d loader."""


import h5py
import MDSplus as mds
import numpy as np
import os
from os.path import join

import logging


# Set the environment variable to connect to the correct MDS server
# 
os.environ["main_path"] = "atlas.gat.com::"

# List of shots, taken from https://nomos.gat.com/DIII-D/physics/miniprop/mp_list.php?mpid=2017-24-89
shot_list = [172337, 172338, 172339, 172340, 172341, 172342, 172342]

# List of diagnostics to fetch. These are taken from https://doi.org/10.1088/1741-4326/abe08d (Abbate et al. 2021)
# and the path names are collected in https://nomos.gat.com/DIII-D/physics/miniprop/mp_list.php?mpid=2017-24-89

# Separate between three kinds of data. Each kind requires separate download logic, 
# to pull from from either MDS or PTDATA, and handle 0d vs 1d
# The first kind are profiles. These are 1d time series.
# Each entry is a dict with 
#   Tree - The name of the MDS tree the data is stored in
#   Node - The name of the MDS node the data is stored in
#   map_to - The group in the HDF5 file the data will be stored in
profile_dict = {'Electron Density': {"Tree": "D3D", "Node": "ELECTRONS.PROFILE_FITS.ZIPFIT:EDENSFIT", "map_to": "edens"},
                "Electron Temp": {"Tree": "D3D", "Node": "ELECTRONS.PROFILE_FITS.ZIPFIT:ETEMPFIT", "map_to": "etemp"},
                "Ion rotation": {"Tree": "ZIPFIT01", "Node": "PROFILES.TROTFIT", "map_to": "ion_rot"},
                "Rot transform": {"Tree": "EFIT01",  "Node": "RESULTS.GEQDSK:QPSI", "map_to": "qpsi"},
                "Plasma pressure": {"Tree": "EFIT01", "Node": "RESULTS.GEQDSK:PRES", "map_to": "pressure"}}

# The second kind are scalar time series, i.e. 0d time series.
# Each entry is a dict with 
#   Tree - The name of the MDS tree the data is stored in
#   Node - The name of the MDS node the data is stored in
#   map_to - The group in the HDF5 file the data will be stored in
scalars_dict = {"tinj": {"Tree": "D3D", "Node": "NB:TINJ", "map_to": "tinj"},
                "pinj": {"Tree": "D3D", "Node": "NB:PINJ", "map_to": "pinj"},
                "Total ECH power": {"Tree": "D3D", "Node": "RF.ECH.TOTAL.ECHPWRC", "map_to": "echpwrc"},
                "pinjf_15l": {"Tree": "D3D", "Node": "NB.NB15L:PINJF_15L", "map_to": "pinjf_15l"},
                "pinjf_15r": {"Tree": "D3D", "Node": "NB.NB15R:PINJF_15R", "map_to": "pinjf_15r"},
                "pinjf_21l": {"Tree": "D3D", "Node": "NB.NB21L:PINJF_21L", "map_to": "pinjf_21l"},
                "pinjf_21r": {"Tree": "D3D", "Node": "NB.NB21R:PINJF_21R", "map_to": "pinjf_21r"},
                "pinjf_30l": {"Tree": "D3D", "Node": "NB.NB30L:PINJF_30L", "map_to": "pinjf_30l"},
                "pinjf_30r": {"Tree": "D3D", "Node": "NB.NB30R:PINJF_30R", "map_to": "pinjf_30r"},
                "pinjf_33l": {"Tree": "D3D", "Node": "NB.NB33L:PINJF_33L", "map_to": "pinjf_33l"},
                "pinjf_33r": {"Tree": "D3D", "Node": "NB.NB33R:PINJF_33R", "map_to": "pinjf_33r"},
                "tecef01": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF01", "map_to": "tecef01"},
                "tecef02": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF02", "map_to": "tecef02"},
                "tecef03": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF03", "map_to": "tecef03"},
                "tecef04": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF04", "map_to": "tecef04"},
                "tecef05": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF05", "map_to": "tecef05"},
                "tecef06": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF06", "map_to": "tecef06"},
                "tecef07": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF07", "map_to": "tecef07"},
                "tecef08": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF08", "map_to": "tecef08"},
                "tecef09": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF09", "map_to": "tecef09"},
                "tecef10": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF10", "map_to": "tecef10"},
                "tecef11": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF11", "map_to": "tecef11"},
                "tecef12": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF12", "map_to": "tecef12"},
                "tecef13": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF13", "map_to": "tecef13"},
                "tecef14": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF14", "map_to": "tecef14"},
                "tecef15": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF15", "map_to": "tecef15"},
                "tecef16": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF16", "map_to": "tecef16"},
                "tecef17": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF17", "map_to": "tecef17"},
                "tecef18": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF18", "map_to": "tecef18"},
                "tecef19": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF19", "map_to": "tecef19"},
                "tecef20": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF20", "map_to": "tecef20"},
                "tecef21": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF21", "map_to": "tecef21"},
                "tecef22": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF22", "map_to": "tecef22"},
                "tecef23": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF23", "map_to": "tecef23"},
                "tecef24": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF24", "map_to": "tecef24"},
                "tecef25": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF25", "map_to": "tecef25"},
                "tecef26": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF26", "map_to": "tecef26"},
                "tecef27": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF27", "map_to": "tecef27"},
                "tecef28": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF28", "map_to": "tecef28"},
                "tecef29": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF29", "map_to": "tecef29"},
                "tecef30": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF30", "map_to": "tecef30"},
                "tecef31": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF31", "map_to": "tecef31"},
                "tecef32": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF32", "map_to": "tecef32"},
                "tecef33": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF33", "map_to": "tecef33"},
                "tecef34": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF34", "map_to": "tecef34"},
                "tecef35": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF35", "map_to": "tecef35"},
                "tecef36": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF36", "map_to": "tecef36"},
                "tecef37": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF37", "map_to": "tecef37"},
                "tecef38": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF38", "map_to": "tecef38"},
                "tecef39": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF39", "map_to": "tecef39"},
                "tecef40": {"Tree": "D3D", "Node": "ELECTRONS.ECE.TECEF.TECEF40", "map_to": "tecef40"},
                "Neutron rate": {"Tree": "D3D", "Node": "IONS.NEUTRONS.FIP:NEUTRONSRATE", "map_to": "neutronsrate", "doc": "https://diii-d.gat.com/diii-d/Diag/neutrons"},
                "Top Triangularity": {"Tree": "EFIT01", "Node": "RESULTS.AEQDSK:DOUTU", "map_to": "doutu"},
                "Bottom Triangularity": {"Tree": "EFIT01", "Node": "RESULTS.AEQDSK:DOUTL", "map_to": "doutl"},
                "Plasma Elongation": {"Tree": "EFIT01", "Node": "RESULTS.AEQDSK:ELONGM", "map_to": "elongm"},
                "Plasma volume": {"Tree": "EFIT01", "Node": "RESULTS.AEQDSK:VOUT", "map_to": "vout"},
                "Internal inductance": {"Tree": "EFIT01", "Node": "RESULTS.AEQDSK:ALI", "map_to": "ali"}}

# The third kind are scalar time series we pull from PTDATA.
# Each entry is a dict with 
#   Node - The name of the MDS node the data is stored in
#   map_to - The group in the HDF5 file the data will be stored in
scalars_pt = {"Target current": {"Node": r"\iptipp", "map_to": "iptipp"},
              "Target density": {"Node": r"\dstdenp", "map_to": "dstdenp"},
              "line-averaged density": {"Node": r"\dssdenest", "map_to": "dssdenest"}}


shotnr = shot_list[0]
data_path = "/projects/EKOLEMEN/d3dloader/test"

log_fname = join(data_path, f"d3d_loader_{shotnr}.log")
logging.basicConfig(filename=log_fname, 
                    format="%(asctime)s    %(message)s",
                    encoding="utf-8", 
                    level=logging.INFO)

conn = mds.Connection("atlas.gat.com")

# File mode needs to be append! Otherwise we delete the file contents every time we
# execute this script.
with h5py.File(join(data_path, f"{shotnr}.h5"), "a") as df:
    assert(df.mode == "r+")
    # Handle each of the three data kinds separately.
    # First profile data
    for key in profile_dict.keys():
        tree = profile_dict[key]["Tree"]
        node = profile_dict[key]["Node"]
        map_to = profile_dict[key]["map_to"]

        try:
            if df[map_to]["zdata"].size > 0:
                logging.info(f"Signal {map_to} already exists. Skipping download")
                continue
        except KeyError as err:
            pass

        try:
            logging.info(f"Trying to download {tree}::{node} from MDS")
            conn.openTree(tree, shotnr)
            zdata = conn.get(f"_s ={node}").data()
            zunits = conn.get('units_of(_s)').data()
            logging.info(f"Downloaded zdata. shape = {zdata.shape}, units = {zunits}")

            xdata = conn.get('dim_of(_s)').data()
            xunits = conn.get('units_of(dim_of(_s))').data()
            if zunits in ["", " "]:
                xunits = conn.get('units(dim_of(_s))').data()
            logging.info(f"Downloaded xdata. shape = {xdata.shape}, units = {xunits}")

            ydata = conn.get('dim_of(_s,1)').data()
            yunits = conn.get('units_of(dim_of(_s,1))').data()
            if yunits in ["", " "]:
                yunits = conn.get('units(dim_of(_s))').data()

            logging.info(f"Downloaded ydata. shape = {ydata.shape}, units = {yunits}")
        except Exception as err:
            logging.error(f"Failed to download {tree}::{node} from MDS: {err}")
            continue
        # The data is downloaded now. Next store them in HDF5
        try:
            grp = df.create_group(map_to)
            grp.attrs.create("origin", f"MDS {tree}::{node}")
            # Store data in arrays and set units as an attribute
            for ds_name, ds_data, u_name, u_data in zip(["xdata", "ydata", "zdata"],
                                                        [xdata, ydata, zdata],
                                                        ["xunits", "yunits", "zunits"],
                                                        [xunits, yunits, zunits]):
                dset = grp.create_dataset(ds_name, ds_data.shape, dtype='f')
                dset[:] = ds_data[:]
                dset.attrs.create(u_name, u_data.encode())
        
        except Exception as err:
            logging.error(f"Failed to write {tree}::{node} to HDF5 group {grp} - {err}")
            raise(err)
        
        logging.info(f"Stored {tree}::{node} into {grp}")

    # Second scalar data
    for key in scalars_dict.keys():
        tree = scalars_dict[key]["Tree"]
        node = scalars_dict[key]["Node"]
        map_to = scalars_dict[key]["map_to"]

        # Skip the download if there already is data in the HDF5 file
        try:
            if df[map_to]["zdata"].size > 0:
                logging.info(f"Signal {map_to} already exists. Skipping download")
                continue
        except KeyError:
            pass

        try:
            logging.info(f"Trying to download {tree}::{node} from MDS")
            conn.openTree(tree, shotnr)

            zdata = conn.get(f"_s ={node}").data()
            zunits = conn.get('units_of(_s)').data()
            logging.info(f"Downloaded zdata. shape={zdata.shape}")

            xdata = conn.get('dim_of(_s)').data()
            xunits = conn.get('units_of(dim_of(_s))').data()
            logging.info(f"Downloaded xdata. shape={xdata.shape}")
        except Exception as err:
            logging.error(f"Failed to download {tree}::{node} from MDS - {err}")
            continue
        
        # Data is now downloaded. Store them in HDF5
        try:
            grp = df.create_group(map_to)
            grp.attrs.create("origin", f"MDS {tree}::{node}")
            # Store data in arrays and set units as an attribute
            for ds_name, ds_data, u_name, u_data in zip(["xdata", "zdata"],
                                                        [xdata, zdata],
                                                        ["xunits", "zunits"],
                                                        [xunits, zunits]):
                dset = grp.create_dataset(ds_name, ds_data.shape, dtype='f')
                dset[:] = ds_data[:]
                dset.attrs.create(u_name, u_data.encode())      
        except Exception as err:
            logging.error(f"Failed to write {tree}::{node} to HDF5 group {grp} - {err}")
            raise(err)
        
        logging.info(f"Stored {tree}::{node} into {grp}")

    # Finally PTDATA
    for key in scalars_pt.keys():
        node = scalars_pt[key]["Node"]
        map_to = scalars_pt[key]["map_to"]
        # Skip the download if there already is data in the HDF5 file
        try:
            if df[map_to]["zdata"].size > 0:
                logging.info(f"Signal {map_to} already exists. Skipping download")
                continue
        except KeyError:
            pass

        try:
            logging.info(f"Trying to download {node} from PTDATA")
            zdata = conn.get(f"_s = ptdata2('{node}', {shotnr})").data()
            xdata = conn.get("dim_of(_s)")
            logging.info(f"Downloaded zdata. shape={zdata.shape}")
            logging.info(f"Downloaded xdata. shape={xdata.shape}")
        except Exception as err:
            logging.error(f"Failed to download {node} from PTDATA - {err}")
            continue

        # Data is downloaded. Store them in HDF5
        try:
            grp = df.create_group(f"{scalars_pt[key]['map_to']}")
            grp.attrs.create("origin", f"PTDATA {node}")
            for ds_name, ds_data in zip(["xdata", "zdata"],
                                        [xdata, zdata]):
                dset = grp.create_dataset(ds_name, ds_data.shape, dtype='f')
                dset[:] = ds_data[:]
        except Exception as err:
            logging.error(f"Failed to write {node} to HDF5 group {grp} - {err}")
            raise(err)
        
        logging.info(f"Stored PTDATA {node} into {grp}")


# end of file downloading.py