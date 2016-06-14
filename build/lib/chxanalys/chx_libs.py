from databroker import DataBroker as db, get_images, get_table, get_events, get_fields
from filestore.api import register_handler, deregister_handler
#from filestore.retrieve import _h_registry, _HANDLER_CACHE, HandlerBase
from eiger_io.pims_reader import EigerImages
from chxtools import handlers
from filestore.path_only_handlers import RawHandler 
## Import all the required packages for  Data Analysis

#* scikit-beam - data analysis tools for X-ray science 
#    - https://github.com/scikit-beam/scikit-beam
#* xray-vision - plotting helper functions for X-ray science
#    - https://github.com/Nikea/xray-vision

import xray_vision
import xray_vision.mpl_plotting as mpl_plot  
from xray_vision.mpl_plotting import speckle
from xray_vision.mask.manual_mask import ManualMask

import skbeam.core.roi as roi
import skbeam.core.correlation as corr
import skbeam.core.utils as utils

import numpy as np
from datetime import datetime
import h5py
 
import pims
from pandas import DataFrame
import os, sys, time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
 

from lmfit import  Model

 
from matplotlib import gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable


import itertools 
mcolors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k','darkgoldenrod','oldlace', 'brown','dodgerblue'   ])
markers = itertools.cycle(list(plt.Line2D.filled_markers))
lstyles = itertools.cycle(['-', '--', '-.','.',':'])






