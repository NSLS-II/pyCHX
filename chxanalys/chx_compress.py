import os,shutil
from glob import iglob

import matplotlib.pyplot as plt
from chxanalys.chx_libs import (np, roi, time, datetime, os,  getpass, db, 
                                      get_images,LogNorm, RUN_GUI)
from chxanalys.chx_generic_functions import (create_time_slice,get_detector, get_fields, get_sid_filenames,  
     load_data)


import struct    
from tqdm import tqdm
from contextlib import closing

from multiprocessing import Pool
import dill
import sys
import gc
import pickle as pkl
# imports handler from CHX
# this is where the decision is made whether or not to use dask
from chxtools.handlers import EigerImages, EigerHandler


def run_dill_encoded(what):    
    fun, args = dill.loads(what)
    return fun(*args)

def apply_async(pool, fun, args, callback=None):    
    return pool.apply_async( run_dill_encoded, (dill.dumps((fun, args)),), callback= callback)


def map_async(pool, fun, args ):    
    return pool.map_async(run_dill_encoded,  (dill.dumps((fun, args)),))

 
def pass_FD(FD,n):
    #FD.rdframe(n)
    FD.seekimg(n)

def go_through_FD(FD):    
    for i in range(FD.beg, FD.end):
        pass_FD(FD,i)
        
    
def compress_eigerdata( images, mask, md, filename=None,  force_compress=False, 
                        bad_pixel_threshold=1e15, bad_pixel_low_threshold=0, 
                       hot_pixel_threshold=2**30, nobytes=4,bins=1, bad_frame_list=None,
                       para_compress= False, num_sub=100, dtypes='uid',reverse =True,
                      num_max_para_process=500, with_pickle=False, direct_load_data=False, data_path=None):   
    
    end= len(images)//bins
    
    if filename is None:
        filename=  '/XF11ID/analysis/Compressed_Data' +'/uid_%s.cmp'%md['uid']
        
    if dtypes!= 'uid':
        para_compress= False  
    else:
        if para_compress:
            images='foo'
            #para_compress=   True
    
    #print( dtypes )
        
    if force_compress:
        print ("Create a new compress file with filename as :%s."%filename)
        if para_compress:
            # stop connection to be before forking... (let it reset again)
            db.reg.disconnect()
            db.mds.reset_connection()
            print( 'Using a multiprocess to compress the data.')
            return para_compress_eigerdata( images, mask, md, filename, 
                        bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                                    bad_pixel_low_threshold=bad_pixel_low_threshold,nobytes= nobytes, bins=bins,
                                     num_sub=num_sub, dtypes=dtypes, reverse=reverse,
                                          num_max_para_process=num_max_para_process, with_pickle= with_pickle,
                                           direct_load_data= direct_load_data,data_path=data_path) 
                    
        else:
            return init_compress_eigerdata( images, mask, md, filename, 
                        bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                                    bad_pixel_low_threshold=bad_pixel_low_threshold,nobytes= nobytes, bins=bins,with_pickle= with_pickle, direct_load_data= direct_load_data,data_path=data_path )        
    else:
        if not os.path.exists( filename ):
            print ("Create a new compress file with filename as :%s."%filename)
            if para_compress:
                print( 'Using a multiprocess to compress the data.')
                return para_compress_eigerdata( images, mask, md, filename, 
                        bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                                    bad_pixel_low_threshold=bad_pixel_low_threshold,nobytes= nobytes, bins=bins,
                                     num_sub=num_sub, dtypes=dtypes, reverse=reverse,
                                              num_max_para_process=num_max_para_process,with_pickle= with_pickle, direct_load_data= direct_load_data,data_path=data_path)
            else:
                return init_compress_eigerdata( images, mask, md, filename, 
                       bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                      bad_pixel_low_threshold=bad_pixel_low_threshold,    nobytes= nobytes, bins=bins,with_pickle= with_pickle, direct_load_data= direct_load_data,data_path=data_path  )  
        else:      
            print ("Using already created compressed file with filename as :%s."%filename)
            beg=0            
            return read_compressed_eigerdata( mask, filename, beg, end, 
                            bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                       bad_pixel_low_threshold=bad_pixel_low_threshold ,bad_frame_list=bad_frame_list,with_pickle= with_pickle, direct_load_data= direct_load_data,data_path=data_path    )  


        
def read_compressed_eigerdata( mask, filename, beg, end,
                              bad_pixel_threshold=1e15, hot_pixel_threshold=2**30,
                             bad_pixel_low_threshold=0,bad_frame_list=None,with_pickle= False,
                             direct_load_data=False,data_path=None):   
    '''
        Read already compress eiger data           
        Return 
            mask
            avg_img
            imsum
            bad_frame_list
            
    ''' 
    #should use try and except instead of with_pickle in the future!
    CAL = False
    if not with_pickle:
        CAL = True
    else:
        try:
            mask, avg_img, imgsum, bad_frame_list_ =  pkl.load( open(filename + '.pkl', 'rb' ) )             
        except:
            CAL = True
    if CAL:            
        FD = Multifile( filename, beg, end)    
        imgsum  =  np.zeros(   FD.end- FD.beg, dtype= np.float  )     
        avg_img = np.zeros(  [FD.md['ncols'], FD.md['nrows'] ] , dtype= np.float )     
        imgsum, bad_frame_list_ = get_each_frame_intensityc( FD, sampling = 1, 
                bad_pixel_threshold=bad_pixel_threshold, bad_pixel_low_threshold=bad_pixel_low_threshold,
                                        hot_pixel_threshold=hot_pixel_threshold, plot_ = False,
                                                bad_frame_list=bad_frame_list) 
        avg_img = get_avg_imgc( FD,  beg=None,end=None,sampling = 1, plot_ = False,bad_frame_list=bad_frame_list_ )
        FD.FID.close() 
            
    return   mask, avg_img, imgsum, bad_frame_list_

def para_compress_eigerdata(  images, mask, md, filename, num_sub=100,
                        bad_pixel_threshold=1e15, hot_pixel_threshold=2**30, 
                            bad_pixel_low_threshold=0, nobytes=4, bins=1, dtypes='uid',reverse =True,
                           num_max_para_process=500, cpu_core_number=72, with_pickle=True,
                           direct_load_data=False, data_path=None):
    
    if dtypes=='uid':
        uid= md['uid'] #images
        if not direct_load_data:
            detector = get_detector( db[uid ] )
            images_ = load_data( uid, detector, reverse= reverse    )
        else:
            images_ = EigerImages(data_path, md)
        N= len(images_)
    
    else:
        N = len(images)    
    N = int( np.ceil( N/ bins  ) )        
    Nf  =  int( np.ceil( N/ num_sub  ) )  
    if Nf >   cpu_core_number:
        print("The process number is larger than %s (XF11ID server core number)"%cpu_core_number)
        num_sub_old = num_sub
        num_sub = int( np.ceil(N/cpu_core_number))
        Nf  =  int( np.ceil( N/ num_sub  ) )
        print ("The sub compressed file number was changed from %s to %s"%( num_sub_old, num_sub ))        
    create_compress_header( md, filename +'-header', nobytes, bins  )    
    #print( 'done for header here')    
    results = para_segment_compress_eigerdata( images=images, mask=mask, md=md,filename=filename, 
                    num_sub=num_sub, bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                    bad_pixel_low_threshold=bad_pixel_low_threshold,nobytes=nobytes, bins=bins, dtypes=dtypes,
                                             num_max_para_process=num_max_para_process,
                                            direct_load_data=direct_load_data, data_path=data_path)
    
    res_ = np.array( [ results[k].get() for k in   list(sorted(results.keys()))   ]     ) 
    imgsum = np.zeros( N )
    bad_frame_list = np.zeros( N, dtype=bool )
    good_count = 1
    for i in range( Nf ):    
        mask_, avg_img_, imgsum_, bad_frame_list_ = res_[i]
        imgsum[i*num_sub: (i+1)*num_sub] = imgsum_
        bad_frame_list[i*num_sub: (i+1)*num_sub] = bad_frame_list_
        if i==0:
            mask = mask_
            avg_img = np.zeros_like( avg_img_ )                  
        else:
            mask *= mask_        
        if not np.sum( np.isnan( avg_img_)):        
            avg_img += avg_img_  
            good_count += 1
            
    bad_frame_list = np.where(  bad_frame_list )[0]    
    avg_img /= good_count   
    
    if len(bad_frame_list):
        print ('Bad frame list are: %s' %bad_frame_list)
    else:
        print ('No bad frames are involved.')    
    print( 'Combining the seperated compressed files together...')
    combine_compressed( filename, Nf, del_old=True) 
    
    del results
    del res_
    if  with_pickle:
        pkl.dump( [mask, avg_img, imgsum, bad_frame_list], open(filename + '.pkl', 'wb' ) )
    return   mask, avg_img, imgsum, bad_frame_list

def combine_compressed( filename,  Nf, del_old=True):
    old_files = np.concatenate(  np.array([ [filename +'-header'], 
                                            [filename  + '_temp-%i.tmp'%i for i in range(Nf) ]]))    
    combine_binary_files(filename, old_files, del_old  ) 
    
def  combine_binary_files(filename, old_files, del_old = False):
    '''Combine binary files together'''
    fn_ = open(filename, 'wb')
    for ftemp in old_files: 
        shutil.copyfileobj( open(ftemp, 'rb'), fn_) 
        if del_old:
            os.remove( ftemp )
    fn_.close() 
    
def para_segment_compress_eigerdata( images, mask,  md, filename, num_sub=100,
                        bad_pixel_threshold=1e15, hot_pixel_threshold=2**30, 
                            bad_pixel_low_threshold=0, nobytes=4, bins=1,  dtypes='images',reverse =True,
                                   num_max_para_process=50,direct_load_data=False, data_path=None):    
    '''
    parallelly compressed eiger data without header, this function is for parallel compress
    ''' 
    if dtypes=='uid':
        uid= md['uid'] #images
        if not direct_load_data:
            detector = get_detector( db[uid ] )
            images_ = load_data( uid, detector, reverse= reverse    )
        else:
            images_ = EigerImages(data_path, md)
        N= len(images_)
    
    else:
        N = len(images) 
   
    #N = int( np.ceil( N/ bins  ) )
    num_sub *= bins    
    if N%num_sub:
        Nf = N// num_sub +1
        print('The average image intensity would be slightly not correct, about 1% error.')
        print( 'Please give a num_sub to make reminder of Num_images/num_sub =0 to get a correct avg_image')
    else:
        Nf = N//num_sub
    print( 'It will create %i temporary files for parallel compression.'%Nf)

    if Nf> num_max_para_process:
        N_runs = np.int( np.ceil( Nf/float(num_max_para_process)))  
        print('The parallel run number: %s is larger than num_max_para_process: %s'%(Nf, num_max_para_process ))
    else:
        N_runs= 1        
    result = {}   
    #print( mask_filename )# + '*'* 10 + 'here' )
    print("start")
    for nr in range( N_runs ):
        if (nr+1)*num_max_para_process > Nf:
            inputs= range( num_max_para_process*nr, Nf )
        else:            
            inputs= range( num_max_para_process*nr, num_max_para_process*(nr + 1 ) )          
        fns = [  filename + '_temp-%i.tmp'%i for i in inputs]
        #print( nr, inputs, )
        pool =  Pool(processes= len(inputs) ) #, maxtasksperchild=1000 )      
        #print( inputs )        
        for i in  inputs:
            if i*num_sub <= N:
                result[i] = pool.apply_async(  segment_compress_eigerdata, [
                    images,  mask,  md,  filename + '_temp-%i.tmp'%i,bad_pixel_threshold, hot_pixel_threshold,    bad_pixel_low_threshold, nobytes, bins, i*num_sub, (i+1)*num_sub, dtypes, reverse,direct_load_data, data_path  ]  )  
       
        pool.close()
        pool.join()
        pool.terminate() 
    print("done")
    return result     

def segment_compress_eigerdata( images,  mask, md, filename, 
                        bad_pixel_threshold=1e15, hot_pixel_threshold=2**30, 
                            bad_pixel_low_threshold=0, nobytes=4, bins=1, 
                               N1=None, N2=None, dtypes='images',reverse =True,direct_load_data=False, data_path=None    ):     
    '''
    Create a compressed eiger data without header, this function is for parallel compress
    for parallel compress don't pass any non-scalar parameters
    '''     
    # TODO : move this hack elsewhere
    #from databroker import Broker
    #db = Broker.named('chx')
    
    if dtypes=='uid':
        uid= md['uid'] #images
        if not direct_load_data:
            detector = get_detector( db[uid ] )
            images = load_data( uid, detector, reverse= reverse    )[N1:N2] 
        else:
            images = EigerImages(data_path, md)[N1:N2] 
    Nimg_ = len( images)     
    M,N = images[0].shape
    avg_img = np.zeros( [M,N], dtype= np.float )    
    Nopix =  float( avg_img.size )
    n=0
    good_count = 0
    #frac = 0.0
    if nobytes==2:
        dtype= np.int16
    elif nobytes==4:
        dtype= np.int32
    elif nobytes==8:
        dtype=np.float64
    else:
        print ( "Wrong type of nobytes, only support 2 [np.int16] or 4 [np.int32]")
        dtype= np.int32   
     
    
    #Nimg =   Nimg_//bins 
    Nimg = int( np.ceil( Nimg_ / bins  ) )
    time_edge = np.array(create_time_slice( N= Nimg_, 
                                    slice_num= Nimg, slice_width= bins ))
    #print( time_edge, Nimg_, Nimg, bins, N1, N2 )
    imgsum  =  np.zeros(    Nimg   )         
    if bins!=1:
        #print('The frames will be binned by %s'%bins) 
        dtype=np.float64
        
    fp = open( filename,'wb' )    
    for n in   range(Nimg):            
        t1,t2 = time_edge[n]  
        if bins!=1:
            img = np.array( np.average(  images[t1:t2], axis=0   )   , dtype= dtype)
        else:
            img =   np.array( images[t1], dtype=dtype) 
        mask &= img < hot_pixel_threshold         
        p = np.where( (np.ravel(img)>0) *  np.ravel(mask) )[0] #don't use masked data 
        v = np.ravel( np.array( img, dtype= dtype )) [p] 
        dlen = len(p)         
        imgsum[n] = v.sum()  
        if (dlen==0) or (imgsum[n] > bad_pixel_threshold) or (imgsum[n] <=bad_pixel_low_threshold):
            dlen = 0
            fp.write(  struct.pack( '@I', dlen  ))    
        else:              
            np.ravel( avg_img )[p] +=   v            
            good_count +=1             
            fp.write(  struct.pack( '@I', dlen   ))
            fp.write(  struct.pack( '@{}i'.format( dlen), *p))
            if bins==1:
                fp.write(  struct.pack( '@{}{}'.format( dlen,'ih'[nobytes==2]), *v)) 
            else:
                fp.write(  struct.pack( '@{}{}'.format( dlen,'dd'[nobytes==2]  ), *v))        #n +=1
        del  p,v, img 
        fp.flush()         
    fp.close()      
    avg_img /= good_count 
    bad_frame_list =  (np.array(imgsum) > bad_pixel_threshold) | (np.array(imgsum) <= bad_pixel_low_threshold)
    sys.stdout.write('#')    
    sys.stdout.flush()    
    #del  images, mask, avg_img, imgsum, bad_frame_list
    #print( 'Should release memory here')    
    return   mask, avg_img, imgsum, bad_frame_list
        
    
        
def create_compress_header( md, filename, nobytes=4, bins=1  ):
    '''
    Create the head for a compressed eiger data, this function is for parallel compress
    '''    
    fp = open( filename,'wb' )
    #Make Header 1024 bytes   
    #md = images.md
    if bins!=1:
        nobytes=8
        
    Header = struct.pack('@16s8d7I916x',b'Version-COMP0001',
                        md['beam_center_x'],md['beam_center_y'], md['count_time'], md['detector_distance'],
                        md['frame_time'],md['incident_wavelength'], md['x_pixel_size'],md['y_pixel_size'],
                        nobytes, md['pixel_mask'].shape[1], md['pixel_mask'].shape[0],
                         0, md['pixel_mask'].shape[1],
                         0, md['pixel_mask'].shape[0]                
                    )
      
    fp.write( Header)       
    fp.close()     
        


def init_compress_eigerdata( images, mask, md, filename, 
                        bad_pixel_threshold=1e15, hot_pixel_threshold=2**30, 
                            bad_pixel_low_threshold=0,nobytes=4, bins=1, with_pickle=True,
                           direct_load_data=False, data_path=None):    
    '''
        Compress the eiger data 
        
        Create a new mask by remove hot_pixel
        Do image average
        Do each image sum
        Find badframe_list for where image sum above bad_pixel_threshold
        Generate a compressed data with filename
        
        if bins!=1, will bin the images with bin number as bins
    
        Header contains 1024 bytes ['Magic value', 'beam_center_x', 'beam_center_y', 'count_time', 'detector_distance', 
           'frame_time', 'incident_wavelength', 'x_pixel_size', 'y_pixel_size', 
           bytes per pixel (either 2 or 4 (Default)),
           Nrows, Ncols, Rows_Begin, Rows_End, Cols_Begin, Cols_End ]
           
        Return 
            mask
            avg_img
            imsum
            bad_frame_list
            
    ''' 
    fp = open( filename,'wb' )
    #Make Header 1024 bytes   
    #md = images.md
    if bins!=1:
        nobytes=8
        
    Header = struct.pack('@16s8d7I916x',b'Version-COMP0001',
                        md['beam_center_x'],md['beam_center_y'], md['count_time'], md['detector_distance'],
                        md['frame_time'],md['incident_wavelength'], md['x_pixel_size'],md['y_pixel_size'],
                        nobytes, md['pixel_mask'].shape[1], md['pixel_mask'].shape[0],
                         0, md['pixel_mask'].shape[1],
                         0, md['pixel_mask'].shape[0]                
                    )
      
    fp.write( Header)  
    
    Nimg_ = len( images)
    avg_img = np.zeros_like(    images[0], dtype= np.float ) 
    Nopix =  float( avg_img.size )
    n=0
    good_count = 0
    frac = 0.0
    if nobytes==2:
        dtype= np.int16
    elif nobytes==4:
        dtype= np.int32
    elif nobytes==8:
        dtype=np.float64
    else:
        print ( "Wrong type of nobytes, only support 2 [np.int16] or 4 [np.int32]")
        dtype= np.int32        
        
        
    Nimg =   Nimg_//bins 
    time_edge = np.array(create_time_slice( N= Nimg_, 
                                    slice_num= Nimg, slice_width= bins ))    
    
    imgsum  =  np.zeros(    Nimg   )         
    if bins!=1:
        print('The frames will be binned by %s'%bins) 
     
    for n in  tqdm( range(Nimg) ):            
        t1,t2 = time_edge[n]
        img = np.average(  images[t1:t2], axis=0   )        
        mask &= img < hot_pixel_threshold   
        p = np.where( (np.ravel(img)>0) &  np.ravel(mask) )[0] #don't use masked data  
        v = np.ravel( np.array( img, dtype= dtype )) [p]
        dlen = len(p)         
        imgsum[n] = v.sum()
        if (imgsum[n] >bad_pixel_threshold) or (imgsum[n] <=bad_pixel_low_threshold):
        #if imgsum[n] >=bad_pixel_threshold :
            dlen = 0
            fp.write(  struct.pack( '@I', dlen  ))    
        else:      
            np.ravel(avg_img )[p] +=   v
            good_count +=1 
            frac += dlen/Nopix
            #s_fmt ='@I{}i{}{}'.format( dlen,dlen,'ih'[nobytes==2])
            fp.write(  struct.pack( '@I', dlen   ))
            fp.write(  struct.pack( '@{}i'.format( dlen), *p))
            if bins==1:
                fp.write(  struct.pack( '@{}{}'.format( dlen,'ih'[nobytes==2]), *v)) 
            else:
                fp.write(  struct.pack( '@{}{}'.format( dlen,'dd'[nobytes==2]  ), *v)) 
        #n +=1     
        
    fp.close() 
    frac /=good_count
    print( "The fraction of pixel occupied by photon is %6.3f%% "%(100*frac) ) 
    avg_img /= good_count
    
    bad_frame_list = np.where( (np.array(imgsum) > bad_pixel_threshold) | (np.array(imgsum) <= bad_pixel_low_threshold)  )[0]
    #bad_frame_list1 = np.where( np.array(imgsum) > bad_pixel_threshold  )[0]
    #bad_frame_list2 = np.where( np.array(imgsum) < bad_pixel_low_threshold  )[0]
    #bad_frame_list =   np.unique( np.concatenate( [bad_frame_list1, bad_frame_list2]) )
    
    
    if len(bad_frame_list):
        print ('Bad frame list are: %s' %bad_frame_list)
    else:
        print ('No bad frames are involved.')
    if  with_pickle:
        pkl.dump( [mask, avg_img, imgsum, bad_frame_list], open(filename + '.pkl', 'wb' ) )    
    return   mask, avg_img, imgsum, bad_frame_list
        

 
"""    Description:

    This is code that Mark wrote to open the multifile format
    in compressed mode, translated to python.
    This seems to work for DALSA, FCCD and EIGER in compressed mode.
    It should be included in the respective detector.i files
    Currently, this refers to the compression mode being '6'
    Each file is image descriptor files chunked together as follows:
            Header (1024 bytes)
    |--------------IMG N begin--------------|
    |                   Dlen
    |---------------------------------------|
    |       Pixel positions (dlen*4 bytes   |
    |      (0 based indexing in file)       |
    |---------------------------------------|
    |    Pixel data(dlen*bytes bytes)       |
    |    (bytes is found in header          |
    |    at position 116)                   |
    |--------------IMG N end----------------|
    |--------------IMG N+1 begin------------|
    |----------------etc.....---------------|

     
     Header contains 1024 bytes version name, 'beam_center_x', 'beam_center_y', 'count_time', 'detector_distance', 
           'frame_time', 'incident_wavelength', 'x_pixel_size', 'y_pixel_size', 
           bytes per pixel (either 2 or 4 (Default)),
           Nrows, Ncols, Rows_Begin, Rows_End, Cols_Begin, Cols_End, 
           
         

"""


class Multifile:
    '''The class representing the multifile.
        The recno is in 1 based numbering scheme (first record is 1)
	This is efficient for reading in increasing order.
	Note: reading same image twice in a row is like reading an earlier
	numbered image and means the program starts for the beginning again. 
 
    '''
    def __init__(self,filename,beg,end):
        '''Multifile initialization. Open the file.
            Here I use the read routine which returns byte objects
            (everything is an object in python). I use struct.unpack
            to convert the byte object to other data type (int object
            etc)
            NOTE: At each record n, the file cursor points to record n+1
        '''
        self.FID = open(filename,"rb")
#        self.FID.seek(0,os.SEEK_SET)
        self.filename = filename
        #br: bytes read
        br = self.FID.read(1024)
        self.beg=beg
        self.end=end
        ms_keys = ['beam_center_x', 'beam_center_y', 'count_time', 'detector_distance', 
           'frame_time', 'incident_wavelength', 'x_pixel_size', 'y_pixel_size',
           'bytes', 
            'nrows', 'ncols', 'rows_begin', 'rows_end', 'cols_begin', 'cols_end'       
          ] 
                
        magic = struct.unpack('@16s', br[:16])
        md_temp =  struct.unpack('@8d7I916x', br[16:])
        self.md = dict(zip(ms_keys, md_temp))
        
        self.imgread=0
        self.recno = 0
        # some initialization stuff        
        self.byts = self.md['bytes']
        if (self.byts==2):
            self.valtype = np.uint16
        elif (self.byts == 4):
            self.valtype = np.uint32
        elif (self.byts == 8): 
            self.valtype = np.float64
        #now convert pieces of these bytes to our data
        self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]        
        
        # now read first image
        #print "Opened file. Bytes per data is {0img.shape = (self.rows,self.cols)}".format(self.byts)

    def _readHeader(self):
        self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]    

    def _readImageRaw(self):
        
        p= np.fromfile(self.FID, dtype = np.int32,count= self.dlen)
        v= np.fromfile(self.FID, dtype = self.valtype,count= self.dlen)
        self.imgread=1
        return(p,v)

    def _readImage(self):
        (p,v)=self._readImageRaw()
        img = np.zeros( (  self.md['ncols'], self.md['nrows']   ) )
        np.put( np.ravel(img), p, v )         
        return(img)

    def seekimg(self,n=None):

        '''Position file to read the nth image.
            For now only reads first image ignores n
        '''
        # the logic involving finding the cursor position      
        if (n is None):
            n = self.recno
        if (n < self.beg or n > self.end):            
            raise IndexError('Error, record out of range')        
        #print (n, self.recno, self.FID.tell() )        
        if ((n == self.recno)  and (self.imgread==0)):            
            pass # do nothing         
            
        else:
            if (n <= self.recno): #ensure cursor less than search pos
                self.FID.seek(1024,os.SEEK_SET)
                self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]
                self.recno = 0
                self.imgread=0
                if n == 0:
                    return 
            #have to iterate on seeking since dlen varies
            #remember for rec recno, cursor is always at recno+1
            if(self.imgread==0 ): #move to next header if need to 
                self.FID.seek(self.dlen*(4+self.byts),os.SEEK_CUR)                
            for i in range(self.recno+1,n):
                #the less seeks performed the faster
                #print (i)
                self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]
                #print 's',self.dlen
                self.FID.seek(self.dlen*(4+self.byts),os.SEEK_CUR)

            # we are now at recno in file, read the header and data
            #self._clearImage()
            self._readHeader()
            self.imgread=0
            self.recno = n
    def rdframe(self,n):
        if self.seekimg(n)!=-1:
            return(self._readImage())

    def rdrawframe(self,n):
        if self.seekimg(n)!=-1:
             return(self._readImageRaw())



            
def pass_FD(FD,n):
    #FD.rdframe(n)
    FD.seekimg(n)            
  



class Multifile_Bins( object  ):
    '''
    Bin a compressed file with bins number
    See Multifile for details for Multifile_class
    '''
    def __init__(self, FD, bins=100):
        '''
        FD: the handler of a compressed Eiger frames
        bins: bins number
       '''          
        
        self.FD=FD
        if (FD.end - FD.beg)%bins:
            print ('Please give a better bins number and make the length of FD/bins= integer')
        else:    
            self.bins = bins
            self.md = FD.md
            #self.beg = FD.beg  
            self.beg = 0
            Nimg = (FD.end - FD.beg)
            slice_num =   Nimg//bins 
            self.end =  slice_num        
            self.time_edge = np.array(create_time_slice( N= Nimg, 
                                    slice_num= slice_num, slice_width= bins )) + FD.beg
            self.get_bin_frame()
        
    def get_bin_frame(self): 
        FD= self.FD
        self.frames = np.zeros( [ FD.md['ncols'],FD.md['nrows'], len(self.time_edge)] )
        for n in  tqdm( range(len(self.time_edge))):
            #print (n)
            t1,t2 = self.time_edge[n]
            #print( t1, t2)
            self.frames[:,:,n] = get_avg_imgc( FD, beg=t1,end=t2, sampling = 1,
                                              plot_ = False, show_progress = False )
    def rdframe(self,n):
        return self.frames[:,:,n]

    def rdrawframe(self,n): 
        x_= np.ravel(  self.rdframe(n) )
        p=  np.where( x_ ) [0]
        v =  np.array( x_[ p ])
        return ( np.array(p, dtype=np.int32), v)

def get_avg_imgc( FD,  beg=None,end=None, sampling = 100, plot_ = False, bad_frame_list=None,
                 show_progress=True, *argv,**kwargs):   
    '''Get average imagef from a data_series by every sampling number to save time'''
    #avg_img = np.average(data_series[:: sampling], axis=0)
    
    if beg is None:
        beg = FD.beg
    if end is None:
        end = FD.end
        
    avg_img = FD.rdframe(beg)
    n=1  
    flag=True    
    if show_progress:        
        #print(  sampling-1 + beg , end, sampling )    
        if bad_frame_list is None:
            bad_frame_list =[]
        fra_num = int( (end -  beg )/sampling ) - len( bad_frame_list )            
        for  i in tqdm(range( sampling-1 + beg , end, sampling  ), desc= 'Averaging %s images'% fra_num):  
            if bad_frame_list is not None:
                if i in bad_frame_list:
                    flag= False
                else:
                    flag=True                  
            #print(i, flag)
            if flag:
                (p,v) = FD.rdrawframe(i)   
                if len(p)>0:
                    np.ravel(avg_img )[p] +=   v
                    n += 1  
    else:
        for  i in range( sampling-1 + beg , end, sampling  ):  
            if bad_frame_list is not None:
                if i in bad_frame_list:
                    flag= False
                else:
                    flag=True 
            if flag:
                (p,v) = FD.rdrawframe(i)
                if len(p)>0:
                    np.ravel(avg_img )[p] +=   v
                    n += 1          
            
    avg_img /= n  
    if plot_:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid']             
        im = ax.imshow(avg_img , cmap='viridis',origin='lower',
                   norm= LogNorm(vmin=0.001, vmax=1e2))
        #ax.set_title("Masked Averaged Image")
        ax.set_title('uid= %s--Masked-Averaged-Image-'%uid)
        fig.colorbar(im)
        if save:
            #dt =datetime.now()
            #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
            path = kwargs['path'] 
            if 'uid' in kwargs:
                uid = kwargs['uid']
            else:
                uid = 'uid'
            #fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'     
            fp = path + "uid=%s--avg-img-"%uid  + '.png'    
            plt.savefig( fp, dpi=fig.dpi)        
        #plt.show() 
    return avg_img

 

def mean_intensityc(FD, labeled_array,  sampling=1, index=None, multi_cor = False):
    """Compute the mean intensity for each ROI in the compressed file (FD), support parallel computation

    Parameters
    ----------
    FD: Multifile class
        compressed file
    labeled_array : array
        labeled array; 0 is background.
        Each ROI is represented by a nonzero integer. It is not required that
        the ROI labels are contiguous
    index : int, list, optional
        The ROI's to use. If None, this function will extract averages for all
        ROIs

    Returns
    -------
    mean_intensity : array
        The mean intensity of each ROI for all `images`
        Dimensions:
            len(mean_intensity) == len(index)
            len(mean_intensity[0]) == len(images)
    index : list
        The labels for each element of the `mean_intensity` list
    """
    
    qind, pixelist = roi.extract_label_indices(  labeled_array  ) 
    
    if labeled_array.shape != ( FD.md['ncols'],FD.md['nrows']):
        raise ValueError(
            " `image` shape (%d, %d) in FD is not equal to the labeled_array shape (%d, %d)" %( FD.md['ncols'],FD.md['nrows'], labeled_array.shape[0], labeled_array.shape[1]) )
    # handle various input for `index`
    if index is None:
        index = list(np.unique(labeled_array))
        index.remove(0)
    else:
        try:
            len(index)
        except TypeError:
            index = [index]

        index = np.array( index )
        #print ('here')
        good_ind  = np.zeros( max(qind), dtype= np.int32 )
        good_ind[ index -1  ]   =    np.arange( len(index) ) +1 
        w =  np.where( good_ind[qind -1 ]   )[0] 
        qind = good_ind[ qind[w] -1 ]
        pixelist = pixelist[w]

          
    # pre-allocate an array for performance
    # might be able to use list comprehension to make this faster
    
    mean_intensity = np.zeros(   [ int( ( FD.end - FD.beg)/sampling ) , len(index)] )    
    #fra_pix = np.zeros_like( pixelist, dtype=np.float64)
    timg = np.zeros(    FD.md['ncols'] * FD.md['nrows']   , dtype=np.int32   ) 
    timg[pixelist] =   np.arange( 1, len(pixelist) + 1  ) 
    #maxqind = max(qind)    
    norm = np.bincount( qind  )[1:]
    n= 0         
    #for  i in tqdm(range( FD.beg , FD.end )):     
    if not multi_cor:
        for  i in tqdm(range( FD.beg, FD.end, sampling  ), desc= 'Get ROI intensity of each frame' ):    
            (p,v) = FD.rdrawframe(i)
            w = np.where( timg[p] )[0]
            pxlist = timg[  p[w]   ] -1 
            mean_intensity[n] = np.bincount( qind[pxlist], weights = v[w], minlength = len(index)+1 )[1:]
            n +=1     
    else:
        ring_masks = [   np.array(labeled_array==i, dtype = np.int64)  for i in np.unique( labeled_array )[1:] ]
        inputs = range( len(ring_masks) )   
        go_through_FD(FD)
        pool =  Pool(processes= len(inputs) )         
        print( 'Starting assign the tasks...')    
        results = {}         
        for i in tqdm ( inputs ): 
            results[i] = apply_async( pool,  _get_mean_intensity_one_q, ( FD, sampling,   ring_masks[i] ) )
        pool.close()    
        print( 'Starting running the tasks...')    
        res =   [ results[k].get() for k in   tqdm( list(sorted(results.keys())) )   ] 
        #return res     
        for i in inputs:            
            mean_intensity[:,i] = res[i]
        print( 'ROI mean_intensit calculation is DONE!')
        del results
        del res    
        
    mean_intensity /= norm        
    return mean_intensity, index


def _get_mean_intensity_one_q( FD, sampling,   labels ):
    mi = np.zeros(     int( ( FD.end - FD.beg)/sampling )   ) 
    n=0    
    qind, pixelist = roi.extract_label_indices(  labels  )    
    # iterate over the images to compute multi-tau correlation 
    fra_pix = np.zeros_like( pixelist, dtype=np.float64)    
    timg = np.zeros(    FD.md['ncols'] * FD.md['nrows']   , dtype=np.int32   ) 
    timg[pixelist] =   np.arange( 1, len(pixelist) + 1  )     
    for  i in range( FD.beg, FD.end, sampling  ):    
        (p,v) = FD.rdrawframe(i)
        w = np.where( timg[p] )[0]
        pxlist = timg[  p[w]   ] -1 
        mi[n] = np.bincount( qind[pxlist], weights = v[w], minlength = 2 )[1:]
        n +=1 
    return mi
            
            

def get_each_frame_intensityc( FD, sampling = 1, 
                             bad_pixel_threshold=1e10, bad_pixel_low_threshold=0, 
                              hot_pixel_threshold=2**30,                             
                             plot_ = False, bad_frame_list=None, save=False, *argv,**kwargs):   
    '''Get the total intensity of each frame by sampling every N frames
       Also get bad_frame_list by check whether above  bad_pixel_threshold  
       
       Usuage:
       imgsum, bad_frame_list = get_each_frame_intensity(good_series ,sampling = 1000, 
                                bad_pixel_threshold=1e10,  plot_ = True)
    '''
    
    #print ( argv, kwargs )
    #mask &= img < hot_pixel_threshold  
    imgsum  =  np.zeros(  int(  (FD.end - FD.beg )/ sampling )  ) 
    n=0
    for  i in tqdm(range( FD.beg, FD.end, sampling  ), desc= 'Get each frame intensity' ): 
        (p,v) = FD.rdrawframe(i)
        if len(p)>0:
            imgsum[n] = np.sum(  v )
        n += 1
    
    if plot_:
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid']        
        fig, ax = plt.subplots()  
        ax.plot(   imgsum,'bo')
        ax.set_title('uid= %s--imgsum'%uid)
        ax.set_xlabel( 'Frame_bin_%s'%sampling )
        ax.set_ylabel( 'Total_Intensity' )
        
        if save:
            #dt =datetime.now()
            #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
            path = kwargs['path'] 
            if 'uid' in kwargs:
                uid = kwargs['uid']
            else:
                uid = 'uid'
            #fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'     
            fp = path + "uid=%s--imgsum-"%uid  + '.png'    
            fig.savefig( fp, dpi=fig.dpi)
        
        plt.show()
       
    bad_frame_list_ = np.where(   ( np.array(imgsum) > bad_pixel_threshold ) | ( np.array(imgsum) <= bad_pixel_low_threshold)  )[0] +  FD.beg  
    
    if  bad_frame_list is not None:    
        bad_frame_list = np.unique(   np.concatenate([bad_frame_list, bad_frame_list_]) )
    else:
        bad_frame_list = bad_frame_list_  
    
    if len(bad_frame_list):
        print ('Bad frame list length is: %s' %len(bad_frame_list))
    else:
        print ('No bad frames are involved.')
    return imgsum,bad_frame_list




