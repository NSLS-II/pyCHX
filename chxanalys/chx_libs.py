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


markers =  {'*': 'star', '2': 'tri_up',  'o': 'circle', '_': 'hline', '.': 'point', 'd': 'thin_diamond', 
                     '4': 'tri_right',   'x': 'x',  '+': 'plus', '<': 'triangle_left',
                     '|': 'vline', '8': 'octagon',  's': 'square', 
                     'p': 'pentagon', ',': 'pixel', '^': 'triangle_up', 'D': 'diamond', 
                     'H': 'hexagon2', '3': 'tri_left', '>': 'triangle_right', 'h': 'hexagon1',
                     'v': 'triangle_down', '1': 'tri_down'}

markers = np.array( [ i for i in markers.keys()] *10 )
colors = np.array( ['darkorange', 'mediumturquoise', 'seashell', 'mediumaquamarine', 'darkblue', 
           'yellowgreen', 'cyan', 'mintcream', 'royalblue', 'springgreen', 'slategray',
           'yellow', 'slateblue', 'darkslateblue', 'papayawhip', 'bisque', 'firebrick', 
           'burlywood', 'gold', 'dodgerblue', 'dimgrey', 'chartreuse', 'deepskyblue', 'honeydew', 
           'orchid', 'red', 'teal', 'steelblue', 'plum', 'limegreen', 'antiquewhite', 
           'linen', 'saddlebrown', 'grey', 'khaki', 'violet', 'hotpink', 'darkslategray', 
           'forestgreen', 'blue', 'lightsalmon', 'turquoise', 'navajowhite', 'peachpuff',
           'greenyellow', 'darkgrey', 'darkkhaki', 'slategrey', 'indigo',
           'darkolivegreen', 'aquamarine', 'moccasin', 'beige', 'ivory', 'olivedrab',
           'whitesmoke', 'paleturquoise', 'blueviolet', 'tomato', 'aqua', 'palegoldenrod', 
           'cornsilk', 'navy', 'mediumvioletred', 'palevioletred', 'aliceblue', 'azure', 
           'snow', 'orangered', 'lightgrey', 'lightpink', 'orange', 'lightsage', 'wheat', 
           'darkorchid', 'mediumslateblue', 'lightslategray', 'green', 'lawngreen', 'tan', 
           'mediumseagreen', 'darksalmon', 'pink', 'oldlace', 'sienna', 'dimgray', 'fuchsia',
           'lemonchiffon', 'purple', 'maroon', 'salmon', 'gainsboro', 'indianred', 'crimson',
           'olive', 'mistyrose', 'lime', 'lightblue', 'darkgreen', 'lightgreen', 'deeppink', 
           'palegreen', 'thistle', 'lightcoral', 'lightgray', 'lightskyblue', 'mediumspringgreen', 
           'mediumblue', 'peru', 'lightgoldenrodyellow', 'darkseagreen', 'mediumorchid', 
           'coral', 'lightyellow', 'chocolate', 'lavenderblush', 'darkred', 'lightseagreen', 
           'darkviolet', 'lightcyan', 'cadetblue', 'blanchedalmond', 'midnightblue', 
           'darksage', 'lightsteelblue', 'darkcyan', 'floralwhite', 'darkgray', 'magenta',
           'lavender', 'sandybrown', 'cornflowerblue', 'sage',  'gray', 
           'mediumpurple', 'lightslategrey', 'powderblue', 'brown', 'seagreen', 'skyblue',
           'silver', 'darkmagenta', 'darkslategrey', 'darkgoldenrod', 'rosybrown', 
           'goldenrod',   'darkturquoise', 'black'] *10 )

colors_ = itertools.cycle(   colors[::-1] )
#colors_ = itertools.cycle(sorted_colors_ )
markers_ = itertools.cycle( markers )



