from numpy import pi,sin,arctan,sqrt,mgrid,where,shape,exp,linspace,std,arange
from numpy import power,log,log10,array,zeros,ones,reshape,mean,histogram,round,int_
from numpy import indices,hypot,digitize,ma,histogramdd,apply_over_axes,sum
from numpy import around,intersect1d, ravel, unique,hstack,vstack,zeros_like
from numpy import save, load, dot
from numpy.linalg import lstsq
from numpy import polyfit,poly1d;
import sys

import matplotlib.pyplot as plt 
#from Init_for_Timepix import * # the setup file 
import time
 
import struct
import numpy as np    
from tqdm import tqdm    
import pandas as pds
from pyCHX.chx_libs import multi_tau_lags



def compress_timepixeldata( p,t, tbins, filename,  md= None,nobytes=2   ):    
    '''
        Compress the timepixeldata
        md: a string to describle the data info
 
            
    ''' 
    fp = open( filename,'wb' )
 
    if md is None:
        md={}
        md['beam_center_x'] = 0
        md['beam_center_y'] = 0
        md['count_time'] = 0
        md['detector_distance'] = 0
        md['frame_time'] = 0
        md['incident_wavelength'] =0
        md['x_pixel_size'] =25
        md['y_pixel_size'] =25
        #nobytes = 2
        md['sx'] = 256
        md['sy'] = 256    
                            
    Header = struct.pack('@16s8d7I916x',b'Version-COMP0001',
                    md['beam_center_x'],md['beam_center_y'], md['count_time'], md['detector_distance'],
                    md['frame_time'],md['incident_wavelength'], md['x_pixel_size'],md['y_pixel_size'],
                        nobytes, md['sy'], md['sx'],
                         0, md['sy'],
                         0, md['sx']                
                    )
        
        
    #Header = struct.pack('@16s%ds%dx'%(n,lx),b'Version-COMP0002', md.encode('ascii')  )      
    fp.write( Header)
     
    tx = np.arange(  t.min(), t.max(), tbins   )  
    N = len(tx)
    print( 'Will Compress %s Frames.'%(N-1)  )
    #imgsum  =  np.zeros(    N   )   
    for i in tqdm( range(N-1) ):
        ind1 = np.argmin( np.abs( tx[i] - t )  )
        ind2 = np.argmin( np.abs( tx[i+1] - t )  )
        #print( i, ind1, ind2 )
        p_i = p[ind1: ind2]
        pos,val = np.unique( p_i, return_counts= True )
        #print(i,  pos, val )            
        dlen = len(pos)         
        #imgsum[i] = val.sum()
        fp.write(  struct.pack( '@I', dlen   ))
        fp.write(  struct.pack( '@{}i'.format( dlen), *pos))
        fp.write(  struct.pack( '@{}{}'.format( dlen,'ih'[nobytes==2]), *val)) 
    fp.close()  
    return N -1 
        #return    imgsum

    
    
    
    
class Get_TimePixel_Arrayc(object):
    '''
    a class to get intested pixels from a images sequence, 
    load ROI of all images into memory 
    get_data: to get a 2-D array, shape as (len(images), len(pixellist))
    
    One example:        
        data_pixel =   Get_Pixel_Array( imgsr, pixelist).get_data()
    '''
    
    def __init__(self, pos, hitime, tbins, pixelist, beg=None, end=None, norm=None, detx = 256, dety = 256):
        '''
        indexable: a images sequences
        pixelist:  1-D array, interest pixel list
        '''
        self.hitime = hitime
        self.tbins   = tbins
        self.tx = np.arange(  self.hitime.min(), self.hitime.max(), self.tbins   )  
        N = len(self.tx)
        
        if beg is None:
            beg = 0
        if end is None:
            end = N
            
        self.beg = beg
        self.end = end            
        self.length = self.end - self.beg 
        self.pos = pos
        self.pixelist = pixelist        
        self.norm = norm   
        self.detx = detx
        self.dety = dety

        
        
    def get_data(self ): 
        '''
        To get intested pixels array
        Return: 2-D array, shape as (len(images), len(pixellist))
        '''
        norm = self.norm
        data_array = np.zeros([ self.length,len(self.pixelist)]) 
        print( data_array.shape)
        
        #fra_pix = np.zeros_like( pixelist, dtype=np.float64)
        timg = np.zeros(    self.detx * self.dety, dtype=np.int32   )         
        timg[self.pixelist] =   np.arange( 1, len(self.pixelist) + 1  ) 
        n=0
        tx = self.tx
        N = len(self.tx)
        print( 'Will Compress %s Frames.'%(N-1)  )
        #imgsum  =  np.zeros(    N   )   
        for i in tqdm( range(N-1) ):
            ind1 = np.argmin( np.abs( tx[i] - self.hitime )  )
            ind2 = np.argmin( np.abs( tx[i+1] - self.hitime )  )
            
            #print( i, ind1, ind2 )
            
            p_i = self.pos[ind1: ind2]
            pos,val = np.unique( p_i, return_counts= True )
            
            #print( val.sum() )

            w = np.where( timg[pos] )[0]
            pxlist = timg[  pos[w]   ] -1 
            #print( val[w].sum() )
            #fra_pix[ pxlist] = v[w] 
            if norm is None:
                data_array[n][ pxlist] = val[w] 
            else: 
                data_array[n][ pxlist] = val[w] / norm[pxlist]   #-1.0                    
            n += 1
            
        return data_array     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def apply_timepix_mask( x,y,t, roi ):
    y1,y2, x1,x2  = roi
    w =  (x < x2) & (x >= x1) & (y < y2) & (y >= y1) 
    return x[w],y[w], t[w]

def get_timepixel_data( data_dir, filename):
    '''give a csv file of a timepixel data, return x,y,t (in ps/6.1)
        y1,y2, x1,x2 = roi   
    '''
    data =  pds.read_csv( data_dir + filename )
    return data['Col'], data['Row'], data['GlobalTimeFine'] #*6.1  #in ps   


def get_timepixel_data_from_series( data_dir, filename_prefix, 
                                   total_filenum = 72, colms = int(1e5) ):
    x = np.zeros( total_filenum * colms )
    y = np.zeros( total_filenum * colms )
    t = zeros( total_filenum * colms )
    for n in range( total_filenum):
        filename = filename_prefix + '_%s.csv'%n
        data =  get_timepixel_data( data_dir, filename )
        if n!=total_filenum-1:
            ( x[n*colms: (n+1)*colms  ], y[n*colms: (n+1)*colms  ], t[n*colms: (n+1)*colms  ] )= (
                                data[0], data[1], data[2])
        else:
            #print(  filename_prefix + '_%s.csv'%n )
            ln = len(data[0])
            #print( ln )
            ( x[n*colms: n*colms + ln  ], y[n*colms:  n*colms + ln  ], t[n*colms:  n*colms + ln   ] )= (
                                data[0], data[1], data[2])
            
    return  x[:n*colms + ln] ,y[:n*colms + ln],t[:n*colms + ln]
        
     

def get_timepixel_image( x,y,t, det_shape = [256, 256], delta_time = None   ):
    '''give x,y, t data to get image in a period of delta_time (in second)'''
    t0 = t.min() *6.1
    tm = t.max() *6.1
    
    if delta_time is not None:
        delta_time *=1e12
        if delta_time > tm:
            delta_time = tm            
    else:
        delta_time = tm
    #print( delta_time)
    t_ = t[t<delta_time]
    x_ = x[:len(t_)]
    y_ = y[:len(t_)]
 
    img = np.zeros( det_shape, dtype= np.int32 )
    pixlist = x_*det_shape[0] + y_ 
    his = np.histogram( pixlist, bins= np.arange( det_shape[0]*det_shape[1] +1) )[0] 
    np.ravel( img )[:] = his
    print( 'The max photon count is %d.'%img.max())
    return img
    
def get_his_taus( t,  bin_step, time_unit= 1.0 ):
    '''Get taus and  histrogram of photons
       Parameters:
            t: the time stamp of photon hitting the detector
            bin_step: bin time step, in unit of ms
       Return:
            taus, in ms
            histogram of photons
    
    '''
    
    bins = np.arange(  t.min()*time_unit , t.max()*time_unit, bin_step   )  #1e6 for us
    #print( bins )
    td = np.histogram( t*time_unit, bins=bins )[0] #do histogram
    taus = (bins - bins[0])  
    return taus[1:], td

def get_multi_tau_lags( oned_count, num_bufs=8 ):
    n = len( oned_count ) 
    num_levels = int(np.log( n/(num_bufs-1))/np.log(2) +1) +1
    tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)
    return lag_steps[ lag_steps < n ]

def get_timepixel_multi_tau_g2(  oned_count, num_bufs=8    ):
    n = len( oned_count )
    lag_steps  = get_multi_tau_lags( oned_count, num_bufs )
    g2 = np.zeros( len( lag_steps) )
        
    for tau_ind, tau in enumerate(lag_steps):
        IP=  oned_count[: n - tau]
        IF= oned_count[tau:n ] 
        #print( IP.shape,IF.shape, tau, n )
        g2[tau_ind]=   np.dot( IP, IF )/ ( IP.mean() * IF.mean() * float( n - tau) )    
    return g2  


def get_timepixel_c12( oned_count  ): 
    noframes = len( oned_count)
    oned_count = oned_count.reshape( noframes, 1 )
    return np.dot(   oned_count, oned_count.T)  / oned_count  / oned_count.T  / noframes
            
    

def get_timepixel_g2(  oned_count   ):
    n = len( oned_count )    
    norm = ( np.arange( n, 0, -1) * 
            np.array( [np.average(oned_count[i:]) for i in range( n )] ) *
            np.array( [np.average(oned_count[0:n-i]) for i in range( n )] )           )    
    return np.correlate(oned_count, oned_count, mode = 'full')[-n:]/norm      
    
    
    
    
    
    
    
    
    
    
    
#########################################
T = True
F = False
 

 
    
def read_xyt_frame(  n=1 ):
    ''' Load the xyt txt files:
        x,y is the detector (x,y) coordinates
        t is the time-encoder (when hitting the detector at that (x,y))
        DATA_DIR is the data filefold path
        DataPref is the data prefix
        n is file number
        the data name will be like: DATA_DIR/DataPref_0001.txt
        return the histogram of the hitting event
    '''
    import numpy as np 
    ni = '%04d'%n
    fp = DATA_DIR +   DataPref + '%s.txt'%ni
    data = np.genfromtxt( fp, skiprows=0)[:,2] #take the time encoder
    td = np.histogram( data, bins= np.arange(11810) )[0] #do histogram    
    return td 
 
def readframe_series(n=1 ):
    ''' Using this universe name for all the loading fucntions'''
    return read_xyt_frame( n )
    

class xpcs( object):
    def __init__(self):
        """ DOCUMENT __init__(   )
        the initilization of the XPCS class
          """
        self.version='version_0'
        self.create_time='July_14_2015'
        self.author='Yugang_Zhang@chx11id_nsls2_BNL'
 
    def delays(self,time=1,
               nolevs=None,nobufs=None, tmaxs=None): 
        ''' DOCUMENT delays(time=)
        Using the lev,buf concept, to generate array of time delays
        return array of delays.
        KEYWORD:  time: scale delays by time ( should be time between frames)
                  nolevs: lev (a integer number)
                  nobufs: buf (a integer number)
                  tmax:   the max time in the calculation, usually, the noframes
                  
        '''
         
        if nolevs is None:nolevs=nolev #defined by the set-up file
        if nobufs is None:nobufs=nobuf #defined by the set-up file
        if tmaxs is None:tmaxs=tmax    #defined by the set-up file
        if nobufs%2!=0:print ("nobuf must be even!!!"    )
        dly=zeros( (nolevs+1)*nobufs/2 +1  )        
        dict_dly ={}
        for i in range( 1,nolevs+1):
            if i==1:imin= 1
            else:imin=nobufs/2+1
            ptr=(i-1)*nobufs/2+ arange(imin,nobufs+1)
            dly[ptr]= arange( imin, nobufs+1) *2**(i-1)            
            dict_dly[i] = dly[ptr-1]            
        dly*=time 
        dly = dly[:-1]
        dly_ = dly[: where( dly < tmaxs)[0][-1] +1 ]
        self.dly=dly 
        self.dly_=dly_
        self.dict_dly = dict_dly 
        return dly
 

    def make_qlist(self):
        ''' DOCUMENT make_qlist( )
        Giving the noqs, qstart,qend,qwidth, defined by the set-up file        
        return qradi: a list of q values, [qstart, ...,qend] with length as noqs
               qlist: a list of q centered at qradi with qwidth.
        KEYWORD:  noqs, qstart,qend,qwidth::defined by the set-up file 
        '''  
        qradi = linspace(qstart,qend,noqs)
        qlist=zeros(2*noqs)
        qlist[::2]= round(qradi-qwidth/2)  #render  even value
        qlist[1::2]= round(qradi+(1+qwidth)/2) #render odd value
        qlist[::2]= int_(qradi-qwidth/2)  #render  even value
        qlist[1::2]= int_(qradi+(1+qwidth)/2) #render odd value
        if qlist_!=None:qlist=qlist_
        return qlist,qradi

    def calqlist(self, qmask=None ,  shape='circle' ):
        ''' DOCUMENT calqlist( qmask=,shape=, )
        calculate the equvilent pixel with a shape,
        return
            qind: the index of q
            pixellist: the list of pixle
            nopr: pixel number in each q
            nopixels: total pixel number       
        KEYWORD:
            qmask, a mask file;
            qlist,qradi is calculated by make_qlist()        
            shape='circle', give a circle shaped qlist
            shape='column', give a column shaped qlist
            shape='row', give a row shaped qlist             
        '''
         
        qlist,qradi = self.make_qlist()
        y, x = indices( [dimy,dimx] )  
        if shape=='circle':
            y_= y- ceny +1;x_=x-cenx+1        
            r= int_( hypot(x_, y_)    + 0.5  )
        elif shape=='column': 
            r= x
        elif shape=='row':
            r=y
        else:pass
        r= r.flatten()         
        noqrs = len(qlist)    
        qind = digitize(r, qlist)
        if qmask is  None:
            w_= where( (qind)%2 )# qind should be odd;print 'Yes'
            w=w_[0]
        else:
            a=where( (qind)%2 )[0]            
            b=where(  mask.flatten()==False )[0]
            w= intersect1d(a,b)
        nopixels=len(w)
        qind=qind[w]/2
        pixellist= (   y*dimx +x ).flatten() [w]
        nopr,bins=histogram( qind, bins= range( len(qradi) +1 ))
        return qind, pixellist,nopr,nopixels

    ###########################################################################
    ########for one_time correlation function for xyt frames
    ##################################################################

    def autocor_xytframe(self, n):
        '''Do correlation for one xyt frame--with data name as n '''
        dly_ = xp.dly_ 
        #cal=0
        gg2=zeros( len( dly_) )
        data = read_xyt_frame( n )  #load data
        datm = len(data)
        for tau_ind, tau in enumerate(dly_):
            IP= data[: datm - tau]
            IF= data[tau: datm ]             
            gg2[tau_ind]=   dot( IP, IF )/ ( IP.mean() * IF.mean() * float( datm - tau) )

        return gg2
 

    def autocor(self, noframes=10):
        '''Do correlation for xyt file,
           noframes is the frame number to be correlated
        '''
        start_time = time.time() 
        for n in range(1,noframes +1 ): # the main loop for correlator
            gg2 = self.autocor_xytframe( n )
            if n==1:g2=zeros_like( gg2 ) 
            g2 += (  gg2 - g2 )/ float( n  ) #average  g2
            #print n
            if noframes>10: #print progress...
                if  n %( noframes / 10) ==0:
                    sys.stdout.write("#")
                    sys.stdout.flush()                
        elapsed_time = time.time() - start_time
        print ( 'Total time: %.2f min' %(elapsed_time/60.)         )
        return g2

    
    def plot(self, y,x=None):
        '''a simple plot'''
        if x is None:x=arange( len(y))
        plt.plot(x,y,'ro', ls='-')
        plt.show()

 
    def g2_to_pds(self, dly, g2, tscale = None):
        '''convert g2 to a pandas frame'''        
        if len(g2.shape)==1:g2=g2.reshape( [len(g2),1] )
        tn, qn = g2.shape
        tindex=xrange( tn )
        qcolumns = ['t'] + [ 'g2' ]
        if tscale is None:tscale = 1.0
        g2t = hstack( [dly[:tn].reshape(tn,1) * tscale, g2 ])        
        g2p = pd.DataFrame(data=g2t, index=tindex,columns=qcolumns)         
        return g2p

    def show(self,g2p,title):
        t = g2p.t  
        N = len( g2p )
        ylim = [g2p.g2.min(),g2p[1:N].g2.max()]
        g2p.plot(x=t,y='g2',marker='o',ls='--',logx=T,ylim=ylim);
        plt.xlabel('time delay, ns',fontsize=12)
        plt.title(title)
        plt.savefig( RES_DIR + title +'.png' )       
        plt.show()
    
 

######################################################
 
if False:
    xp=xpcs(); #use the xpcs class
    dly = xp.delays()
    if T:
        fnum = 100
        g2=xp.autocor( fnum )
        filename='g2_-%s-'%(fnum)
        save( RES_DIR + FOUT + filename, g2)
        ##g2= load(RES_DIR + FOUT + filename +'.npy')
        g2p = xp.g2_to_pds(dly,g2, tscale = 20)
        xp.show(g2p,'g2_run_%s'%fnum)
