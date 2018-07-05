from databroker import DataBroker as db
from eiger_io.pims_reader import EigerImages
from chxtools import handlers

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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#%matplotlib notebook
#%matplotlib inline
import scipy.optimize as opt
        
from numpy.fft import fftshift, fft2, ifft2
from skimage.util import crop, pad        
from numpy import log    



from .chx_generic_functions import show_img, plot1D
from .Mask_Maker import load_metadata, load_images




def mkbox(   mask_seg  ):
    
    ''' Crop the non_zeros pixels of an image  to a new image 
        and pad with zeros to make double the new image size
        to avoid periodic artifacts from the FFT at correlation lengths.
    '''
    pxlst = np.where(mask_seg.ravel())[0]
    dims = mask_seg.shape
    imgwidthy = dims[1]   #dimension in y, but in plot being x
    imgwidthx = dims[0]   #dimension in x, but in plot being y
    #x and y are flipped???
    #matrix notation!!!
    pixely = pxlst%imgwidthy
    pixelx = pxlst//imgwidthy

    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)    
    img_crop = crop(  mask_seg, ((minpixelx, imgwidthx -  maxpixelx -1 ),
                                (minpixely, imgwidthy -  maxpixely -1 )) )
    widthx = maxpixelx - minpixelx + 1
    widthy = maxpixely - minpixely + 1
     
    oldimg = np.zeros(dims)
    oldimg.ravel()[pxlst] = 1
    #img_crop = np.copy(oldimg[minpixelx:maxpixelx+1,minpixely:maxpixely+1])
    #subpxlst = np.where(img_crop.ravel() != 0)[0]    
    
    #print("min/maxpixelx: {},{};min/maxpixely: {},{}".format(minpixelx,maxpixelx,minpixely,maxpixely))
    
    #padx = 2**np.int( np.ceil(  log(widthx +20  )/log(2) ) ) - widthx
    #pady = 2**np.int( np.ceil(  log(widthy +20)/log(2) ) ) -widthy
    
    padx = 2**np.int( np.ceil(  log(widthx*3)/log(2) ) ) - widthx
    pady = 2**np.int( np.ceil(  log(widthy*3)/log(2) ) ) -widthy
    
    img_pad = pad( img_crop, ((0, padx),(0, pady ) ) ,'constant', constant_values=(0,0))
    subpxlst = np.where(img_pad.ravel() != 0)[0]    
    return np.array(subpxlst,dtype = np.int32),  np.array(img_pad,dtype = np.int32) 

def cross_corr(img1,img2,mask=None):
    '''Compute the autocorrelation of two images.
        Right now does not take mask into account.
        todo: take mask into account (requires extra calculations)
        input: 
            img1: first image
            img2: second image
            mask: a mask array
        output:
            the autocorrelation of the two images (same shape as the correlated images)
        
    '''
    #if(mask is not None):
    #   img1 *= mask
    #  img2 *= mask
    
    #img1_mean = np.mean( img1.flat )
    #img2_mean = np.mean( img2.flat )
    
    # imgc = fftshift( ifft2(     
    #        fft2(img1/img1_mean -1.0 )*np.conj(fft2( img2/img2_mean -1.0 ))).real )
    
    #imgc = fftshift( ifft2(     
    #        fft2(  img1/img1_mean  )*np.conj(fft2(  img2/img2_mean   ))).real )
    
    imgc = fftshift( ifft2(     
            fft2(  img1  )*np.conj(fft2(  img2  ))).real )
    
    #imgc /= (img1.shape[0]*img1.shape[1])**2
    if(mask is not None):
        maskc = cross_corr(mask,mask)        
        imgc /= np.maximum( 1, maskc )
            
            
    return imgc


def cross_corr_allregion(img1, img2, labeled_array):
    """
    Spatial correlation between all ROI subregions of two images.
    
    Correctly handle irregular masks by carefully zero-padding
    to avoid periodic artifacts from the FFT at correlation
    lengths.
    """    
    result = []
    for label in np.unique(labeled_array):
        mask = labeled_array == label
        res = cross_corr(  img1, img2, mask  )
        result.append(res)
    return result

def cross_corr_subregion(img1, img2, mask_seg, center = None, q_pixel=None,sq = None):
    """
    Spatial correlation between a subregion of two images.
    
    Correctly handle irregular masks by carefully zero-padding
    to avoid periodic artifacts from the FFT at correlation
    lengths.
    
    input: 
        img1: first image
        img2: second image
        mask_seg: a mask array to take the ROI from images
        
        center = None, q_pixel=None,sq =None:  for isotropic samples
        
    output:
        imgc: the autocorrelation of the two images  
        
    """
    pxlst = np.where( mask_seg.ravel())[0]
    
    dims = img1.shape
    imgwidthy = dims[1]   #dimension in y, but in plot being x
    imgwidthx = dims[0]   #dimension in x, but in plot being y
    #x and y are flipped???
    #matrix notation!!! 
    
    subpxlst,submask = mkbox(   mask_seg  )
    subimg1 = np.zeros_like(submask, dtype = float)
    subimg2 = np.zeros_like(submask, dtype=float)
    
    #print (subimg1.shape, subimg2.shape)
    
    if center is not None:
        pixely = pxlst%imgwidthy -center[1]
        pixelx = pxlst//imgwidthy - center[0]
        r= np.int_( np.hypot(pixelx, pixely)    + 0.5  )  #why add 0.5?
        
        subimg1.ravel()[subpxlst] = img1.ravel()[pxlst]/ np.interp( r,q_pixel,sq ) -1.0
        subimg2.ravel()[subpxlst] = img2.ravel()[pxlst]/ np.interp( r,q_pixel,sq )  -1.0
    else:
        subimg1.ravel()[subpxlst] = img1.ravel()[pxlst] 
        subimg2.ravel()[subpxlst] = img2.ravel()[pxlst]   
        
    imgc = cross_corr(subimg1,subimg2, mask=submask)
    # autocorr(img2, img2, mask=mask)
    return imgc#, subimg1, subimg2 



 


import scipy.optimize as opt


def get_cross_plot(  imgc, imgc_width = None, imgc_widthy = None,plot_ = False, cx=None, cy=None ): 
    ''' 
       imgc: the autocorrelation of the two images
             the center part with imgc_width of the correlated result 
    
        Return:
        imgc_center_part, cx, cy (the max intensity center)
    '''
    
    #imgc = cross_corr_subregion(img1, img2, center, mask_seg, q_pixel, sq  )
    cenx,ceny = int(imgc.shape[0]/2), int(imgc.shape[1]/2)
    if imgc_width is None:
        mx,my =   np.where( imgc >1e-10 ) 
        x1,x2,y1,y2 = mx[0], mx[-1], my[0], my[-1]
        imgc_center_part = imgc[x1:x2, y1:y2]
        
    else:
        imgc_widthx = imgc_width
        if imgc_widthy is None:
            imgc_widthy = imgc_width
        imgc_center_part = imgc[ cenx-imgc_widthx:cenx +imgc_widthx,  ceny-imgc_widthy:ceny + imgc_widthy ]
    if cx is  None:
        cx, cy = np.where( imgc_center_part == imgc_center_part.max() )
        cx = cx[0]
        cy = cy[0]
    #print ( 'The pixel with max intensity in X-direction is: %s' % np.argmax ( imgc_center[imgc_w ]) )
    #print ( 'The pixel with max intensity in Y-direction is: %s' % np.argmax ( imgc_center[:,imgc_w ]) )
    
    #print ( 'The pixel shift in X-direction is: %s' % (cx  - imgc_width))
    #print ( 'The pixel shift in Y-direction is: %s' % (cy - imgc_width))
    
    if plot_:
        plot1D( imgc_center_part[ cx] )
    if plot_:
        plot1D( imgc_center_part[:,cy ] )
    
    return imgc_center_part,  cx, cy
    
        
def twoD_Gaussian( xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    Two-D Gaussian Function
    
    '''
    x,y = xy[0], xy[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def fit_two_Gaussian( imgc_center_part,  cx, cy,   initial_guess = (1., 30, 30, 2, 2, 0, 0),plot_=False  ):
    '''
    Fit image by a two-D gaussian function.
    initial_guess =  amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    '''
    
    # Create x and y indices
    
    imgc_w = int( imgc_center_part.shape[0]/2 )
    x = np.linspace(0, imgc_w*2-1,  imgc_w*2 )
    y = np.linspace(0, imgc_w*2-1,  imgc_w*2 )
    x, y = np.meshgrid(x, y)
    #xy = np.array ( (x,y) )
    xy = np.array ( (x.ravel(),y.ravel()) ) 
    popt, pcov = opt.curve_fit(twoD_Gaussian, xy,  imgc_center_part.ravel(), p0=initial_guess)
    data_fitted = twoD_Gaussian( xy, *popt)  #.reshape(imgc_w*2, imgc_w*2)
    #show_img( data_fitted.reshape(imgc_w*2, imgc_w*2)  )
    
    
    kai = np.sum( np.square( data_fitted  - imgc_center_part.ravel() ) )
    
    print ( kai, popt)
    #print ( pcov )
    
    print ( "Area =  %s"%( popt[0]/( 2*np.pi* popt[3]* popt[4 ])  )   )
    #print (  popt[0]/( 2*np.pi* popt[3]* popt[4 ])  )
    
    
    fx_h, fy_h =      np.meshgrid( cx,   np.linspace(0, imgc_w*2-1,  imgc_w*2 *4 )  )    
    fxy_h = np.array ( (fx_h.ravel(),fy_h.ravel()) ) 
    data_fitted_h = twoD_Gaussian( fxy_h, *popt)
    
    
    fx_v, fy_v  =      np.meshgrid( np.linspace(0, imgc_w*2-1,  imgc_w*2 *4 ),       cy )  
    fxy_v = np.array ( (fx_v.ravel(),fy_h.ravel()) ) 
    data_fitted_v = twoD_Gaussian( fxy_v, *popt)
    
    if plot_:
    
        fig = plt.figure( )
        ax = fig.add_subplot( 211 ) 
        ax.set_title('Horzontal Fit') 
        plotx =  np.linspace(0, imgc_w*2-1,  imgc_w*2 *4 )    
        ax.plot(    imgc_center_part[ cx]  , 'bo' )      
        ax.plot(  plotx, data_fitted_h  , 'r-') 


        ax = fig.add_subplot( 212 ) 
        ax.set_title('Vertical Fit') 
        plotx =  np.linspace(0, imgc_w*2-1,  imgc_w*2 *4 )    
        ax.plot(    imgc_center_part[ :,cy]  , 'bo' )      
        ax.plot(  plotx, data_fitted_v  , 'r-') 

    return     
    #plot1D( imgc_center[ cx] )
    #plot1D( imgc_center[:,cy ] )






