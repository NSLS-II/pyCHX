"""
Aug 10, Developed by Y.G.@CHX
yuzhang@bnl.gov
This module is for parallel computation of time correlation
"""
from __future__ import absolute_import, division, print_function

from skbeam.core.utils import multi_tau_lags
from skbeam.core.roi import extract_label_indices

from chxanalys.chx_libs import tqdm
from chxanalys.chx_correlationc import (  get_pixelist_interp_iq, _validate_and_transform_inputs,
                 _one_time_process as _one_time_processp,                       )
from chxanalys.chx_compress import ( run_dill_encoded,apply_async, map_async,pass_FD, go_through_FD  ) 


from multiprocessing import Pool
import dill
from collections import namedtuple

import numpy as np
import skbeam.core.roi as roi
import sys

import logging
logger = logging.getLogger(__name__)





class _internal_statep():
    
    def __init__(self, num_levels, num_bufs, labels):
        
        (label_array, pixel_list, num_rois, num_pixels, lag_steps, buf,
     img_per_level, track_level, cur, norm,
     lev_len) = _validate_and_transform_inputs(num_bufs, num_levels, labels)
    
        G = np.zeros(( int( (num_levels + 1) * num_bufs / 2), num_rois),
                 dtype=np.float64)        
        # matrix for normalizing G into g2
        past_intensity = np.zeros_like(G)
        # matrix for normalizing G into g2
        future_intensity = np.zeros_like(G)

        (self.buf,
        self.G,
        self.past_intensity,
        self.future_intensity,
        self.img_per_level,
        self.label_array,
        self.track_level,
        self.cur,
        self.pixel_list,
        self.num_pixels,
        self.lag_steps,
        self.norm,
        self.lev_len ) =  (buf,
        G,
        past_intensity,
        future_intensity,
        img_per_level,
        label_array,
        track_level,
        cur,
        pixel_list,
        num_pixels,
        lag_steps,
        norm,
        lev_len )

    def __getstate__(self):
        """ This is called before pickling. """
        state = self.__dict__.copy()        
        return state

    def __setstate__(self, state):
        """ This is called while unpickling. """
        self.__dict__.update(state)  
        


def lazy_one_timep(FD, num_levels, num_bufs, labels,
            internal_state=None, bad_frame_list=None, imgsum=None, 
                  norm = None ):
    
    if internal_state is None:
        internal_state = _internal_statep(num_levels, num_bufs, labels)    
    # create a shorthand reference to the results and state named tuple    
    s = internal_state
    
    qind, pixelist = roi.extract_label_indices(  labels  )    
    # iterate over the images to compute multi-tau correlation  
    
    fra_pix = np.zeros_like( pixelist, dtype=np.float64)
    
    timg = np.zeros(    FD.md['ncols'] * FD.md['nrows']   , dtype=np.int32   ) 
    timg[pixelist] =   np.arange( 1, len(pixelist) + 1  ) 
    
    if bad_frame_list is None:
        bad_frame_list=[]
    
    #for  i in tqdm(range( FD.beg , FD.end )):
    for  i in range( FD.beg , FD.end ):
        if i in bad_frame_list:
            fra_pix[:]= np.nan
        else:
            (p,v) = FD.rdrawframe(i)
            w = np.where( timg[p] )[0]
            pxlist = timg[  p[w]   ] -1 
            
            if imgsum is None:
                if norm is None:
                    #print ('here')
                    fra_pix[ pxlist] = v[w] 
                    
                else:                     
                    fra_pix[ pxlist] = v[w]/ norm[pxlist]   #-1.0    
                    
            else:
                if norm is None:
                    fra_pix[ pxlist] = v[w] / imgsum[i] 
                else:
                    fra_pix[ pxlist] = v[w]/ imgsum[i]/  norm[pxlist]           
        
        level = 0   
        # increment buffer
        s.cur[0] = (1 + s.cur[0]) % num_bufs 
        # Put the ROI pixels into the ring buffer. 
        s.buf[0, s.cur[0] - 1] =  fra_pix        
        fra_pix[:]=0
        
        #print( i, len(p), len(w), len( pixelist))
        
        #print ('i= %s init fra_pix'%i )            
        buf_no = s.cur[0] - 1
        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity, future_intensity,
        # and img_per_level in place!
        
        #print (s.G)
        _one_time_processp(s.buf, s.G, s.past_intensity, s.future_intensity,
                          s.label_array, num_bufs, s.num_pixels,
                          s.img_per_level, level, buf_no, s.norm, s.lev_len)

        #print (s.G)
        # check whether the number of levels is one, otherwise
        # continue processing the next level
        processing = num_levels > 1

        level = 1
        
        while processing:
            if not s.track_level[level]:
                s.track_level[level] = True
                processing = False
            else:
                prev = (1 + (s.cur[level - 1] - 2) % num_bufs)
                s.cur[level] = (
                    1 + s.cur[level] % num_bufs)

                s.buf[level, s.cur[level] - 1] = ((
                        s.buf[level - 1, prev - 1] +
                        s.buf[level - 1, s.cur[level - 1] - 1]) / 2)

                # make the track_level zero once that level is processed
                s.track_level[level] = False

                # call processing_func for each multi-tau level greater
                # than one. This is modifying things in place. See comment
                # on previous call above.
                buf_no = s.cur[level] - 1
                _one_time_processp(s.buf, s.G, s.past_intensity,
                                  s.future_intensity, s.label_array, num_bufs,
                                  s.num_pixels, s.img_per_level, level, buf_no,
                                  s.norm, s.lev_len)
                level += 1

                # Checking whether there is next level for processing
                processing = level < num_levels
                

    # If any past intensities are zero, then g2 cannot be normalized at
    # those levels. This if/else code block is basically preventing
    # divide-by-zero errors.
    if len(np.where(s.past_intensity == 0)[0]) != 0:
        g_max = np.where(s.past_intensity == 0)[0][0]
    else:
        g_max = s.past_intensity.shape[0]

    g2 = (s.G[:g_max] / (s.past_intensity[:g_max] *
                         s.future_intensity[:g_max]))
    
    #print (FD)

    #sys.stdout.write('#')
    #del FD
    #sys.stdout.flush()
    #print (g2)
    #return results(g2, s.lag_steps[:g_max], s)
    return g2, s.lag_steps[:g_max] #, s
   

def cal_g2p( FD, ring_mask, bad_frame_list=None, 
            good_start=0, num_buf = 8, num_lev = None, imgsum=None, norm=None ):
    '''calculation g2 by using a multi-tau algorithm
       for a compressed file with parallel calculation
    '''
    FD.beg = max(FD.beg, good_start)
    noframes = FD.end - FD.beg +1   # number of frames, not "no frames"
    for i in range(FD.beg, FD.end):
        pass_FD(FD,i)    
    if num_lev is None:
        num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    print ('In this g2 calculation, the buf and lev number are: %s--%s--'%(num_buf,num_lev))
    if  bad_frame_list is not None:
        if len(bad_frame_list)!=0:
            print ('Bad frame involved and will be precessed!')            
            noframes -=  len(np.where(np.in1d( bad_frame_list, 
                                              range(good_start, FD.end)))[0])   
    print ('%s frames will be processed...'%(noframes))      
    ring_masks = [   np.array(ring_mask==i, dtype = np.int64) 
              for i in np.unique( ring_mask )[1:] ]    
    qind, pixelist = roi.extract_label_indices(  ring_mask  )    
    if norm is not None:
        norms = [ norm[ np.in1d(  pixelist, 
            extract_label_indices( np.array(ring_mask==i, dtype = np.int64))[1])]
                for i in np.unique( ring_mask )[1:] ] 

    inputs = range( len(ring_masks) )    
    
    pool =  Pool(processes= len(inputs) )
    internal_state = None       
    print( 'Starting assign the tasks...')    
    results = {}    
    if norm is not None: 
        for i in  tqdm( inputs ): 
            results[i] =  apply_async( pool, lazy_one_timep, ( FD, num_lev, num_buf, ring_masks[i],
                        internal_state,  bad_frame_list, imgsum,
                                    norms[i], ) ) 
    else:
        #print ('for norm is None')    
        for i in tqdm ( inputs ): 
            results[i] = apply_async( pool, lazy_one_timep, ( FD, num_lev, num_buf, ring_masks[i], 
                        internal_state,   bad_frame_list,imgsum, None,
                                     ) )             
    pool.close()    
    print( 'Starting running the tasks...')    
    res =   [ results[k].get() for k in   tqdm( list(sorted(results.keys())) )   ] 
    len_lag = 10**10
    for i in inputs:  #to get the smallest length of lag_step
        if len_lag > len(  res[i][1]  ):
            lag_steps  = res[i][1]  
            len_lag  =   len(  lag_steps   )            
    #lag_steps  = res[0][1]  
    g2 = np.zeros( [len( lag_steps),len(ring_masks)] )
    
    for i in inputs:
        #print( res[i][0][:,0].shape, g2.shape )
        g2[:,i] = res[i][0][:,0][:len_lag]  
        
    print( 'G2 calculation DONE!')
    del results
    del res
    return  g2, lag_steps  



    
    
    
def auto_two_Arrayp(  data_pixel, rois, index=None):
    
    ''' 
    TODO list
    will try to use dask
    
    Dec 16, 2015, Y.G.@CHX
    a numpy operation method to get two-time correlation function using parallel computation
    
    Parameters:
        data:  images sequence, shape as [img[0], img[1], imgs_length]
        rois: 2-D array, the interested roi, has the same shape as image, can be rings for saxs, boxes for gisaxs
    
    Options:
        
        data_pixel: if not None,    
                    2-D array, shape as (len(images), len(qind)),
                    use function Get_Pixel_Array( ).get_data(  ) to get 
         
   
    Return:
        g12: a 3-D array, shape as ( imgs_length, imgs_length, q)
     
    One example:        
        g12 = auto_two_Array( imgsr, ring_mask, data_pixel = data_pixel ) 
    '''            
    qind, pixelist = roi.extract_label_indices(   rois  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:] 
    noframes = data_pixel.shape[0]    
    g12b = np.zeros(  [noframes, noframes, noqs] )
    
    if index is None:
        index =  np.arange( 1, noqs + 1 )        
    else:
        try:
            len(index)
            index = np.array(  index  )
        except TypeError:
            index = np.array(  [index]  )
    qlist = np.arange( 1, noqs + 1 )[ index -1  ]    
    

    inputs = range( len(qlist) ) 
    
    
    data_pixel_qis = [0]* len(qlist)
    for i in inputs:         
        pixelist_qi =  np.where( qind ==  qlist[i] )[0] 
        data_pixel_qis[i] =    data_pixel[:,pixelist_qi]    
    
    #pool =  Pool(processes= len(inputs) )  
    #results = [ apply_async( pool, _get_two_time_for_one_q, ( qlist[i], 
    #                                    data_pixel_qis[i], nopr, noframes ) ) for i in tqdm( inputs )  ]        
    #res = [r.get() for r in results]    
    
    
    
    pool =  Pool(processes= len(inputs) )  
    results = {}        
    for i in  inputs:        
        results[i] =   pool.apply_async(  _get_two_time_for_one_q, [
                            qlist[i], data_pixel_qis[i], nopr, noframes
                               ]  )  
    pool.close()
    pool.join()     
    res = np.array( [ results[k].get() for k in   list(sorted(results.keys()))   ]     )   
    
    #print('here')
    
    for i in inputs:
        qi=qlist[i]
        g12b[:,:,qi -1 ] = res[i]        
    print( 'G12 calculation DONE!')        
    return g12b #g12b


def _get_two_time_for_one_q( qi, data_pixel_qi, nopr, noframes   ):
    
    #print( data_pixel_qi.shape)
    
    sum1 = (np.average( data_pixel_qi, axis=1)).reshape( 1, noframes   )  
    sum2 = sum1.T 
    two_time_qi = np.dot(   data_pixel_qi, data_pixel_qi.T)  /sum1  / sum2  / nopr[qi -1]
    return  two_time_qi
    