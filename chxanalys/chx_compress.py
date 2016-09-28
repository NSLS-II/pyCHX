import os
import matplotlib.pyplot as plt
from chxanalys.chx_libs import (np, roi, time, datetime, os,  getpass, db, get_images,LogNorm, RUN_GUI)
import struct    
from tqdm import tqdm


def compress_eigerdata( images, mask, md, filename, force_compress=False, 
                        bad_pixel_threshold=1e15, bad_pixel_low_threshold=0, 
                       hot_pixel_threshold=2**30, nobytes=4  ):
    
    
    
    if force_compress:
        print ("Create a new compress file with filename as :%s."%filename)
        return init_compress_eigerdata( images, mask, md, filename, 
                        bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                                    bad_pixel_low_threshold=bad_pixel_low_threshold,nobytes= nobytes  )        
    else:
        if not os.path.exists( filename ):
            print ("Create a new compress file with filename as :%s."%filename)
            return init_compress_eigerdata( images, mask, md, filename, 
                       bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                      bad_pixel_low_threshold=bad_pixel_low_threshold,    nobytes= nobytes )  
        else:      
            print ("Using already created compressed file with filename as :%s."%filename)
            beg=0
            end= len(images)
            return read_compressed_eigerdata( mask, filename, beg, end, 
                            bad_pixel_threshold=bad_pixel_threshold, hot_pixel_threshold=hot_pixel_threshold, 
                       bad_pixel_low_threshold=bad_pixel_low_threshold     )  


        
def read_compressed_eigerdata( mask, filename, beg, end,
                              bad_pixel_threshold=1e15, hot_pixel_threshold=2**30,
                             bad_pixel_low_threshold=0):   
    '''
        Read already compress eiger data           
        Return 
            mask
            avg_img
            imsum
            bad_frame_list
            
    ''' 
    FD = Multifile( filename, beg, end)    
    imgsum  =  np.zeros(   FD.end- FD.beg, dtype= np.float  )     
    avg_img = np.zeros(  [FD.md['ncols'], FD.md['nrows'] ] , dtype= np.float )
    
    avg_img = get_avg_imgc( FD,  beg=None,end=None,sampling = 1, plot_ = False ) 
    
    imgsum, bad_frame_list = get_each_frame_intensityc( FD, sampling = 1, 
            bad_pixel_threshold=bad_pixel_threshold, bad_pixel_low_threshold=bad_pixel_low_threshold,
                                    hot_pixel_threshold=hot_pixel_threshold, plot_ = False)    

    FD.FID.close()
    return   mask, avg_img, imgsum, bad_frame_list

def init_compress_eigerdata( images, mask, md, filename, 
                        bad_pixel_threshold=1e15, hot_pixel_threshold=2**30, 
                            bad_pixel_low_threshold=0,nobytes=4  ):    
    '''
        Compress the eiger data 
        
        Create a new mask by remove hot_pixel
        Do image average
        Do each image sum
        Find badframe_list for where image sum above bad_pixel_threshold
        Generate a compressed data with filename
        
    
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
    Header = struct.pack('@16s8d7I916x',b'Version-COMP0001',
                        md['beam_center_x'],md['beam_center_y'], md['count_time'], md['detector_distance'],
                        md['frame_time'],md['incident_wavelength'], md['x_pixel_size'],md['y_pixel_size'],
                        nobytes, md['pixel_mask'].shape[1], md['pixel_mask'].shape[0],
                         0, md['pixel_mask'].shape[1],
                         0, md['pixel_mask'].shape[0]                
                    )
      
    fp.write( Header)  
    
    imgsum  =  np.zeros(    len( images )  ) 
    avg_img = np.zeros_like(    images[0], dtype= np.float ) 
    Nopix =  float( avg_img.size )
    n=0
    good_count = 0
    frac = 0.0
    if nobytes==2:
        dtype= np.int16
    elif nobytes==4:
        dtype= np.int32
    else:
        print ( "Wrong type of nobytes, only support 2 [np.int16] or 4 [np.int32]")
        dtype= np.int32
        
    for img in tqdm( images ):       
        
        mask &= img < hot_pixel_threshold   
        p = np.where( (np.ravel(img)>0) &  np.ravel(mask) )[0] #don't use masked data        
        v = np.ravel( np.array( img, dtype= dtype )) [p]
        dlen = len(p)         
        imgsum[n] = v.sum()
        
        if imgsum[n] >=bad_pixel_threshold:
            dlen = 0
            fp.write(  struct.pack( '@I', dlen  ))    
        else:      
            np.ravel(avg_img )[p] +=   v
            good_count +=1 
            frac += dlen/Nopix
            #s_fmt ='@I{}i{}{}'.format( dlen,dlen,'ih'[nobytes==2])
            fp.write(  struct.pack( '@I', dlen   ))
            fp.write(  struct.pack( '@{}i'.format( dlen), *p))
            fp.write(  struct.pack( '@{}{}'.format( dlen,'ih'[nobytes==2]), *v))        
        n +=1      
    fp.close() 
    frac /=good_count
    print( "The fraction of pixel occupied by photon is %6.3f%% "%(100*frac) ) 
    avg_img /= good_count
    
    #bad_frame_list = np.where( (np.array(imgsum) > bad_pixel_threshold) & (np.array(imgsum) < bad_pixel_low_threshold)  )[0]
    bad_frame_list1 = np.where( np.array(imgsum) > bad_pixel_threshold  )[0]
    bad_frame_list2 = np.where( np.array(imgsum) < bad_pixel_low_threshold  )[0]
    bad_frame_list =   np.unique( np.concatenate( [bad_frame_list1, bad_frame_list2]) )
    
    
    if len(bad_frame_list):
        print ('Bad frame list are: %s' %bad_frame_list)
    else:
        print ('No bad frames are involved.')
        
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
  



def get_avg_imgc( FD,  beg=None,end=None,sampling = 100, plot_ = False,  *argv,**kwargs):   
    '''Get average imagef from a data_series by every sampling number to save time'''
    #avg_img = np.average(data_series[:: sampling], axis=0)
    
    if beg is None:
        beg = FD.beg
    if end is None:
        end = FD.end
        
    avg_img = FD.rdframe(beg)
    n=1    
    for  i in tqdm(range( sampling-1 + beg , end, sampling  ), desc= 'Averaging images' ):  
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


def mean_intensityc(FD, labeled_array,  sampling=1, index=None):
    """Compute the mean intensity for each ROI in the compressed file (FD)

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
    for  i in tqdm(range( FD.beg, FD.end, sampling  ), desc= 'Get ROI intensity of each frame' ):    
        (p,v) = FD.rdrawframe(i)
        w = np.where( timg[p] )[0]
        pxlist = timg[  p[w]   ] -1 
        mean_intensity[n] = np.bincount( qind[pxlist], weights = v[w], minlength = len(index)+1 )[1:]
        n +=1
        
    mean_intensity /= norm
        
    return mean_intensity, index





def get_each_frame_intensityc( FD, sampling = 1, 
                             bad_pixel_threshold=1e10, bad_pixel_low_threshold=0, 
                              hot_pixel_threshold=2**30,                             
                             plot_ = False,  *argv,**kwargs):   
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
       
    bad_frame_list = np.where(   ( np.array(imgsum) > bad_pixel_threshold ) | ( np.array(imgsum) <= bad_pixel_low_threshold)  )[0] +  FD.beg  
    
    #|  (np.array(imgsum)==0) )[0] +  FD.beg  
    
    #print (  np.array(imgsum), bad_pixel_threshold,  bad_pixel_low_threshold)
    
    if len(bad_frame_list):
        print ('Bad frame list length is: %s' %len(bad_frame_list))
    else:
        print ('No bad frames are involved.')
    return imgsum,bad_frame_list















