"""
Dec 10, 2015 Developed by Y.G.@CHX 
yuzhang@bnl.gov
This module is for the necessary packages for the XPCS analysis 
"""


from skbeam.core.utils import multi_tau_lags

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
import  getpass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle 

from lmfit import  Model
from lmfit import minimize, Parameters, Parameter, report_fit

from matplotlib.figure import Figure 
from matplotlib import gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import collections
import itertools 


mcolors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k','darkgoldenrod','oldlace', 'brown','dodgerblue'   ])


markers = itertools.cycle(list(plt.Line2D.filled_markers))
lstyles = itertools.cycle(['-', '--', '-.','.',':'])

colors =  itertools.cycle(["blue", "darkolivegreen", "brown", "m", "orange", "hotpink", "darkcyan",  "red",
            "gray", "green", "black", "cyan", "purple" , "navy"])
colors_copy =  itertools.cycle(["blue", "darkolivegreen", "brown", "m", "orange", "hotpink", "darkcyan",  "red",
            "gray", "green", "black", "cyan", "purple" , "navy"])
markers = itertools.cycle( ["o", "2", "p", "1", "s", "*", "4",  "+", "8", "v","3", "D",  "H", "^",])
markers_copy = itertools.cycle( ["o", "2", "p", "1", "s", "*", "4",  "+", "8", "v","3", "D",  "H", "^",])


RUN_GUI = False  #if True for gui setup; else for notebook; the main code difference is the Figure() or plt.figure(figsize=(8, 6))

markers =  ['o',   'H', 'D', 'v',  '^', '<',  '>', 'p',
                's',  'h',   '*', 'd', 
            '$I$','$L$', '$O$','$V$','$E$', 
            '$c$', '$h$','$x$','$b$','$e$','$a$','$m$','$l$','$i$','$n$', '$e$',
            '8', '1', '3', '2', '4',     '+',   'x',  '_',   '|', ',',  '1',]

markers = np.array(   markers *100 )
colors = np.array( ['darkorange', 'mediumturquoise', 'seashell', 'mediumaquamarine', 'darkblue', 
           'yellowgreen',  'mintcream', 'royalblue', 'springgreen', 'slategray',
           'yellow', 'slateblue', 'darkslateblue', 'papayawhip', 'bisque', 'firebrick', 
           'burlywood',  'dodgerblue', 'dimgrey', 'chartreuse', 'deepskyblue', 'honeydew', 
           'orchid',  'teal', 'steelblue', 'limegreen', 'antiquewhite', 
           'linen', 'saddlebrown', 'grey', 'khaki',  'hotpink', 'darkslategray', 
           'forestgreen',  'lightsalmon', 'turquoise', 'navajowhite', 
            'darkgrey', 'darkkhaki', 'slategrey', 'indigo',
           'darkolivegreen', 'aquamarine', 'moccasin', 'beige', 'ivory', 'olivedrab',
           'whitesmoke', 'paleturquoise', 'blueviolet', 'tomato', 'aqua', 'palegoldenrod', 
           'cornsilk', 'navy', 'mediumvioletred', 'palevioletred', 'aliceblue', 'azure', 
             'orangered', 'lightgrey', 'lightpink', 'orange', 'lightsage', 'wheat', 
           'darkorchid', 'mediumslateblue', 'lightslategray', 'green', 'lawngreen', 
           'mediumseagreen', 'darksalmon', 'pink', 'oldlace', 'sienna', 'dimgray', 'fuchsia',
           'lemonchiffon', 'maroon', 'salmon', 'gainsboro', 'indianred', 'crimson',
            'mistyrose', 'lightblue', 'darkgreen', 'lightgreen', 'deeppink', 
           'palegreen', 'thistle', 'lightcoral', 'lightgray', 'lightskyblue', 'mediumspringgreen', 
           'mediumblue', 'peru', 'lightgoldenrodyellow', 'darkseagreen', 'mediumorchid', 
           'coral', 'lightyellow', 'chocolate', 'lavenderblush', 'darkred', 'lightseagreen', 
           'darkviolet', 'lightcyan', 'cadetblue', 'blanchedalmond', 'midnightblue', 
           'darksage', 'lightsteelblue', 'darkcyan', 'floralwhite', 'darkgray', 
           'lavender', 'sandybrown', 'cornflowerblue', 'sage',  'gray', 
           'mediumpurple', 'lightslategrey',   'seagreen', 
           'silver', 'darkmagenta', 'darkslategrey', 'darkgoldenrod', 'rosybrown', 
           'goldenrod',   'darkturquoise', 'plum',
                 'purple',   'olive', 'gold','powderblue',  'peachpuff','violet', 'lime',  'greenyellow', 'tan',    'skyblue',
                    'magenta',   'black', 'brown',   'green', 'cyan', 'red','blue'] *100 )

colors = colors[::-1]

colors_ = itertools.cycle(   colors  )
#colors_ = itertools.cycle(sorted_colors_ )
markers_ = itertools.cycle( markers )    
    
    


