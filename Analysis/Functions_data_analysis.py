#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Jan 16 15:44:10 2023

Functions for analysing MITgcm output. 
Output should be loaded using Functions_load_output and used as inputs here. 

@author: gemma
"""

import xarray as xr
import numpy as np
from scipy import stats, signal
from math import isnan
import math
import bisect
from Functions_load_output import load_experiment_ds, get_bathymetry_and_troughs


# %% Location dictionaries

# Plotting regions for making 2d maps
plot_regions = {'AS_far':[-68,-76.5,-135,-90],\
                'AS_long':[-66,-76.5,-135,-90],\
                'AS_near':[-70,-76.5,-120,-95],\
                'bathy_map':[-70,-76.5,-120,-95],\
                'ASBS':[-68,-76.5,-130,-75],\
                'PIB':[-74.5,-76,-105,-98],\
                'Inner Shelf':[-74,-76,-110,-98],\
                'full_model_domain':[-65.5,-76.5,-140,-65.1]}


# For map region analyses, e.g. shelf-break undercurrent strength, total-shelf means, heat budget. 
analysis_region_dict = {'thesis_shelf_box':[-71.6,-76,-111,-100],
                        'naughten_shelf_box':[-70.8,-76,-115,-100],
                        'shelf_break':[-70.9,-71.9,-115,-100],
                        'inner_shelf_corner':[-75,-76,-109,-98],
                        'PIG_shelf':[-75.2,-75.9,-101.5,-98],
                        'ase_domain':[-70,-76,-115,-100]}
                    # naughten shelf box is a bigger shelf-box that is used to search for shelf-break

