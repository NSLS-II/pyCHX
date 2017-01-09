
######################################################################################
########Dec 16, 2015, Yugang Zhang, yuzhang@bnl.gov, CHX, NSLS-II, BNL################
########Time correlation function, include one-time, two-time, four-time##############
########Muli-tau method, array-operation method#######################################
######################################################################################


import numpy as np
import sys
import time
import skbeam.core.roi as roi
from matplotlib import gridspec

import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
mcolors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k','darkgoldenrod','oldlace', 'brown','dodgerblue'   ])
markers = itertools.cycle(list(plt.Line2D.filled_markers))
lstyles = itertools.cycle(['-', '--', '-.','.',':'])



####################################
##Multi-Tau--One time correlation
#####################################

def autocor_one_time( num_buf,  rois, imgs, num_lev=None, start_img=None, end_img=None,
                    bad_images = None, threshold=None):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    a multi-tau code for one-time correlation function,
    add new funciton to deal with bad images, which masked intensities are still
    large than threshold
    
    Parameters:
        num_buf: int, number of buffer
        rois: 2-D array, the interested roi, has the same shape as image, can be rings for saxs, boxes for gisaxs
        imgs: pims sequences, image stack
    Options:
        num_lev: int, number of level, if None: = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
        start_img: int, None, =0
        end_img: int, None, = len(imgs)
        bad_images: list, None,bad_images list
        threshold: float, None, intensity max threshold, above which is considered as bad images
   
    Return:
    g2, 2D-array, shape as (tau, q)
    tau, 1D-array    
    
    One example:
        
        g2, tau = autocor_one_time( num_buf,  ring_mask, imgsr, num_lev=None,
                                       bad_images=None, threshold= 65500 )
    '''
      
    start_time = time.time()
    #print (dly)
    if start_img is None:
        start_img=0
    if end_img is None:
        try:
            end_img= len(imgs)
        except:
            end_img= imgs.length
            
    #print (start_img, end_img)    
    noframes = end_img - start_img #+ 1
    #print (noframes)
    ring_mask = rois
    if num_lev is None:num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    nolev = num_lev
    nobuf =num_buf
    print ( 'The lev number is %s'%num_lev)
    
    dly, dict_dly = delays( num_lev, num_buf, time=1 )
    #print (dly.max())
    lev_leng = np.array( [  len(  dict_dly[i] ) for i in list(dict_dly.keys())   ])
    
    qind, pixelist = roi.extract_label_indices(   ring_mask  )
    noqs = np.max(qind)    
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]
    nopixels = nopr.sum()     
    start_time = time.time() 
   
    buf =  np.ma.zeros([num_lev,num_buf,nopixels])
    buf.mask = True   
            
    
    cts=np.zeros(num_lev)
    cur=np.ones(num_lev) * num_buf
    countl = np.array( np.zeros(  num_lev ),dtype='int')  
    
    g2 =  np.zeros( [ noframes, noframes, noqs] )   
    
    G=np.zeros( [(nolev+1)*int(nobuf/2),noqs])
    IAP=np.zeros( [(nolev+1)*int(nobuf/2),noqs])
    IAF=np.zeros( [(nolev+1)*int(nobuf/2),noqs])
    num= np.array( np.zeros(  num_lev ),dtype='int')  
    
    Num= { key: [0]* len(  dict_dly[key] ) for key in list(dict_dly.keys())  }
    print ('Doing g2 caculation of %s frames---'%(noframes ))
    ttx=0     
    #if bad_images is None:bad_images=[]
    for n in range( start_img, end_img ):   ##do the work here  
        img = imgs[n] 
        
    #for n, img in enumerate( imgs):
        img_ = (np.ravel(img))[pixelist]
        
        #print ( img_.max() )
        if threshold is not None:
            if img_.max() >= threshold:
                print ('bad image: %s here!'%n)
                img_ =  np.ma.zeros( len(img_) )
                img_.mask = True    
                
        if bad_images is not None:        
            if n in bad_images:
                print ('bad image: %s here!'%n)
                img_ =  np.ma.zeros( len(img_) )
                img_.mask = True 
        
        
        cur[0]=1+cur[0]%num_buf  # increment buffer  
 
        buf[0, cur[0]-1 ]=  img_

        img=[] #//save space 
        img_=[]
        countl[0] = 1+ countl[0]
 
        process_one_time(lev=0, bufno=cur[0]-1,
            G=G,IAP=IAP,IAF=IAF, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly, Num=Num, lev_leng=lev_leng )     
        #time_ind[0].append(  current_img_time   )
        processing=1
        lev=1
        while processing:
            if cts[lev]:
                prev=  1+ (cur[lev-1]-1-1+num_buf)%num_buf
                cur[lev]=  1+ cur[lev]%num_buf
                countl[lev] = 1+ countl[lev] 
 
                bufa = buf[lev-1,prev-1]
                bufb=  buf[lev-1,cur[lev-1]-1] 
                
                if (bufa.data==0).all():
                    buf[lev,cur[lev]-1] =  bufa
                elif (bufb.data==0).all():
                    buf[lev,cur[lev]-1] = bufb 
                else:
                    buf[lev,cur[lev]-1] = ( bufa + bufb ) /2. 
                
                cts[lev]=0                
                t1_idx=   (countl[lev]-1) *2
 
                process_one_time(lev=lev, bufno=cur[lev]-1,
                        G=G,IAP=IAP,IAF=IAF, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly,Num =Num, lev_leng=lev_leng )     
 
                lev+=1
                #//Since this level finished, test if there is a next level for processing
                if lev<num_lev:processing = 1
                else:processing = 0                                
            else:
                cts[lev]=1      #// set flag to process next time
                processing=0    #// can stop until more images are accumulated              
 
        
        if  n %( int(noframes/10) ) ==0:
            sys.stdout.write("#")
            sys.stdout.flush()                
    #print G.shape    
    if (len(np.where(IAP==0)[0])!=0) and ( 0 not in nopr):
        gmax = np.where(IAP==0)[0][0]        
    else:
        gmax=IAP.shape[0]
    #g2=G/(IAP*IAF)
    #print G
    g2=(G[:gmax]/(IAP[:gmax]*IAF[:gmax]))       
    elapsed_time = time.time() - start_time
    #print (Num)
    print ('Total time: %.2f min' %(elapsed_time/60.))        
    return  g2,dly[:gmax]  #, elapsed_time/60.


    
            
def process_one_time(lev, bufno,    
                     G,IAP,IAF, buf, num, num_buf,noqs,qind,nopr, dly,Num,lev_leng ):
    '''
    a function for  autocor_one_time
    '''
    
    num[lev]+=1     
    if lev==0:imin=0
    else:imin= int(num_buf/2 )         
    for i in range(imin, min(num[lev],num_buf) ):        
        ptr=lev*int(num_buf/2)+i    
        delayno=int( (bufno-i)%num_buf) #//cyclic buffers 
        
        IP  =  buf[lev,delayno]
        IF  =  buf[lev,bufno]  
        ind = ptr - lev_leng[:lev].sum()        
        IP_ = IP.copy()
        IF_ = IF.copy()
        
        if (IP.data ==0).all():            
            IF_=np.zeros( IP.shape )
            IP_= np.zeros( IP.shape )
            Num[lev+1][ind] += 1                    
        if (IF.data ==0).all():  
            #print ('here IF =0')
            IF_ =  np.zeros( IF.shape )
            IP_=   np.zeros( IF.shape ) 
            if (IP.data ==0).all():
                pass                
            else:
                Num[lev+1][ind] += 1          
        norm_num = num[lev]-i -  Num[lev+1][ind]  
        
        #print (   lev, ptr, num[lev]-i, Num[lev+1][ind] )
        #print (ind, lev_leng)
        
        if not (IP_ ==0).all():
            G[ptr]+=  (   np.bincount(qind,      weights= IF_*IP_ )[1:]/nopr- G[ptr] )/ norm_num
            IAP[ptr]+= (  np.bincount(qind,   weights= IP_)[1:]/nopr-IAP[ptr] )/ norm_num
            IAF[ptr]+= (  np.bincount(qind,   weights= IF_)[1:]/nopr-IAF[ptr] )/ norm_num
 



####################################
##Multi-Tau--Two time correlation
#####################################


def autocor_two_time( num_buf, rois, imgs, num_lev=None, start_img=None, end_img=None    ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    a multi-tau code for two-time correlation function  
    
    Parameters:
        num_buf: int, number of buffer
        rois: 2-D array, the interested roi, has the same shape as image, can be rings for saxs, boxes for gisaxs
        imgs: pims sequences, image stack
    Options:
        num_lev: int, number of level, if None: = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
        start_img: int, None, =0
        end_img: int, None, = len(imgs)
        #to be done to deal with bad frames
        #bad_images: list, None,bad_images list
        #threshold: float, None, intensity max threshold, above which is considered as bad images
   
    Return:
    g12, 3D-array, shape as ( len(imgs), len(imgs), q)
       
        
    One example:
        
        g12  = autocor_two_time( num_buf,  ring_mask, imgsr, num_lev=None )
    '''
    

    #print (dly)
    if start_img is None:start_img=0
    if end_img is None:
        try:
            end_img= len(imgs)
        except:
            end_img= imgs.length
            
    #print (start_img, end_img)    
    noframes = end_img - start_img #+ 1
    #print (noframes)
    ring_mask = rois
    if num_lev is None:num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    print ( 'The lev number is %s'%num_lev)
    
    dly, dict_dly = delays( num_lev, num_buf, time=1 )
    #print (dly.max())
    
    qind, pixelist = roi.extract_label_indices(   ring_mask  )
    noqs = np.max(qind)    
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]
    nopixels = nopr.sum() 
    
    start_time = time.time()
    
    buf=np.zeros([num_lev,num_buf,nopixels])  #// matrix of buffers, for store img
    
    
    cts=np.zeros(num_lev)
    cur=np.ones(num_lev) * num_buf
    countl = np.array( np.zeros(  num_lev ),dtype='int')  
    
    g12 =  np.zeros( [ noframes, noframes, noqs] )      
    
    num= np.array( np.zeros(  num_lev ),dtype='int')          
    time_ind ={key: [] for key in range(num_lev)}   
    
    ttx=0        
    for n in range( start_img, end_img ):   ##do the work here
        
        cur[0]=1+cur[0]%num_buf  # increment buffer  
        img = imgs[n] 
        
        #print ( 'The insert image is %s' %(n) )
    
        buf[0, cur[0]-1 ]=  (np.ravel(img))[pixelist]
        img=[] #//save space 
        countl[0] = 1+ countl[0]
        current_img_time = n - start_img +1
    
        process_two_time(lev=0, bufno=cur[0]-1,n=current_img_time,
                        g12=g12, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly)     
        time_ind[0].append(  current_img_time   )
        processing=1
        lev=1
        while processing:
            if cts[lev]:
                prev=  1+ (cur[lev-1]-1-1+num_buf)%num_buf
                cur[lev]=  1+ cur[lev]%num_buf
                countl[lev] = 1+ countl[lev]                                
                buf[lev,cur[lev]-1] = ( buf[lev-1,prev-1] + buf[lev-1,cur[lev-1]-1] ) /2.
                cts[lev]=0                
                t1_idx=   (countl[lev]-1) *2
                current_img_time = ((time_ind[lev-1])[t1_idx ] +  (time_ind[lev-1])[t1_idx +1 ] )/2. 
                time_ind[lev].append(  current_img_time      )  
                process_two_time(lev=lev, bufno=cur[lev]-1,n=current_img_time,
                        g12=g12, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly)  
                lev+=1
                #//Since this level finished, test if there is a next level for processing
                if lev<num_lev:processing = 1
                else:processing = 0                                
            else:
                cts[lev]=1      #// set flag to process next time
                processing=0    #// can stop until more images are accumulated              
 
        
        if  n %(noframes/10) ==0:
            sys.stdout.write("#")
            sys.stdout.flush()                
    
    
    for q in range(noqs):            
        x0 =  g12[:,:,q]
        g12[:,:,q] = np.tril(x0) +  np.tril(x0).T - np.diag( np.diag(x0) )            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    
    return g12, elapsed_time/60.

    
            
def process_two_time(lev, bufno,n ,    
                     g12, buf, num, num_buf,noqs,qind,nopr, dly ):
    '''a function for autocor_two_time'''
    num[lev]+=1  
    if lev==0:imin=0
    else:imin= int(num_buf/2 )
    for i in range(imin, min(num[lev],num_buf) ):
        ptr=lev*int(num_buf/2)+i    
        delayno=(bufno-i)%num_buf #//cyclic buffers            
        IP=buf[lev,delayno]
        IF=buf[lev,bufno]
        I_t12 =  (np.histogram(qind, bins=noqs, weights= IF*IP))[0]
        I_t1  =  (np.histogram(qind, bins=noqs, weights= IP))[0]
        I_t2  =  (np.histogram(qind, bins=noqs, weights= IF))[0]
        tind1 = (n-1)
        tind2=(n -dly[ptr] -1)
        
        if not isinstance( n, int ):                
            nshift = 2**(lev-1)                
            for i in range( -nshift+1, nshift +1 ):
                #print tind1+i
                g12[ int(tind1 + i), int(tind2 + i) ] =I_t12/( I_t1 * I_t2) * nopr
        else:
                #print tind1
            g12[ tind1, tind2 ]  =   I_t12/( I_t1 * I_t2) * nopr       
        
        
        
    

####################################
##Array_Operation--Two time correlation
#####################################
   


def auto_two_Array( data, rois, data_pixel=None  ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    a numpy operation method to get two-time correlation function 
    
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
      
        
    start_time = time.time()
     
    qind, pixelist = roi.extract_label_indices(   rois  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]    
     
    if data_pixel is None:
        data_pixel =   Get_Pixel_Array( data, pixelist).get_data()
        #print (data_pixel.shape)
    
    try:
        noframes = len(data)
    except:
        noframes = data.length
    g12b = np.zeros(  [noframes, noframes, noqs] )
    Unitq = (noqs/10)
    proi=0
    
    for qi in range(1, noqs + 1 ):
        pixelist_qi =  np.where( qind == qi)[0] 
        #print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi =    data_pixel[:,pixelist_qi]   
        
        sum1 = (np.average( data_pixel_qi, axis=1)).reshape( 1, noframes   )  
        sum2 = sum1.T       
        
        g12b[:,:,qi -1 ] = np.dot(   data_pixel_qi, data_pixel_qi.T)  /sum1  / sum2  / nopr[qi -1]
        #print ( proi, int( qi //( Unitq) ) )
        if  int( qi //( Unitq) ) == proi:
            sys.stdout.write("#")
            sys.stdout.flush() 
            proi += 1
            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    return g12b


    
def auto_two_Array_g1_norm( data, rois, data_pixel=None  ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    a numpy operation method to get two-time correlation function with a normalization
    the purpose for this fucntion is to get a exactly same result as one-time correlation function
    
    Parameters:
        data:  images sequence, shape as [img[0], img[1], imgs_length]
        rois: 2-D array, the interested roi, has the same shape as image, can be rings for saxs, boxes for gisaxs
    
    Options:
        
        data_pixel: if not None,    
                    2-D array, shape as (len(images), len(qind)),
                    use function Get_Pixel_Array( ).get_data(  ) to get 
         
   
    Return:
        g12b_norm: a 3-D array, shape as ( imgs_length, imgs_length, q), 
                   a convention two-time correlation functio  
                   same as obtained by auto_two_Array( data, roi, data_pixel=None  ) 
        g12b:      a 3-D array, shape as ( imgs_length, imgs_length, q), 
                   a non-normlized two-time correlation function
                   
        norms:     a 2-D array, shape as ( imgs_length,   q), a normalization for further get one-time from two time  
     
    One example:        
        g12b_norm, g12b_not_norm, norms = auto_two_Array_g1_norm( imgsr, ring_mask, data_pixel = data_pixel ) 
        
    '''
        
        
        
    start_time = time.time()
    qind, pixelist = roi.extract_label_indices(   rois  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]    
    if data_pixel is None:
        data_pixel =   Get_Pixel_Array( data, pixelist).get_data()
        #print (data_pixel.shape)
    
    try:
        noframes = len(data)
    except:
        noframes = data.length
    g12b_norm = np.zeros(  [noframes, noframes, noqs] )
    g12b = np.zeros(  [noframes, noframes, noqs] )
    norms = np.zeros(  [noframes, noqs] )
    
    Unitq = (noqs/10)
    proi=0
    
    for qi in range(1, noqs + 1 ):
        pixelist_qi =  np.where( qind == qi)[0] 
        #print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi =    data_pixel[:,pixelist_qi]   
        
        sum1 = (np.average( data_pixel_qi, axis=1)).reshape( 1, noframes   )  
        sum2 = sum1.T        
        #norms_g12  =  sum1 * sum2 * nopr[qi -1]
        norms[:,qi -1 ]  =  sum1 
        
        g12b[:,:,qi -1 ] = np.dot(   data_pixel_qi, data_pixel_qi.T)  
        g12b_norm[:,:,qi -1 ] = g12b[:,:,qi -1 ]/ sum1 / sum2 / nopr[qi -1]
        #print ( proi, int( qi //( Unitq) ) )
        if  int( qi //( Unitq) ) == proi:
            sys.stdout.write("#")
            sys.stdout.flush() 
            proi += 1
            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    return g12b_norm, g12b, norms


####################################
##Derivation of Two time correlation
#####################################
    
#####################################
#get one-time
#####################################
 

def get_one_time_from_two_time(  g12, norms=None, nopr = None   ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Get one-time correlation function from two correlation function
    namely, calculate the mean of each diag line of g12 to get one-time correlation fucntion
    
    Parameters:
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q) 
    
    Options:
        norms: if not None, a 2-D array, shape as ( imgs_length,   q), a normalization for further get one-time from two time, get by:  g12b_norm, g12b_not_norm, norms = auto_two_Array_g1_norm( imgsr, ring_mask, data_pixel = data_pixel ) 
        nopr: if not None, 1-D array, shape as [q], the number of interested pixel of each q 
         
   
    Return:
        g2f12: a 2-D array, shape as ( imgs_length,  q), 
                   a one-time correlation function  
     
    One example:        
        g2b_norm = get_one_time_from_two_time(g12b_norm,  norms=None, nopr=None )
        g2b_not_norm = get_one_time_from_two_time(g12b_not_norm, norms=norms, nopr=nopr)   
    '''    
 
    m,n,noqs = g12.shape
    g2f12 = []       
    for q in  range(noqs):   
        temp=[]    
        y=g12[:,:,q]        
        for tau in range(m): 
            if norms is None:
                temp.append( np.diag(y,k=int(tau)).mean() )
            else:
                yn = norms[:,q]
                 
                yn1 =  np.average( yn[tau:] )
                yn2 =  np.average( yn[: m-tau] )                  
                temp.append(  np.diag(y,k=int(tau)).mean()/  (yn1*yn2*nopr[q])   )
        temp = np.array( temp).reshape( len(temp),1)
        if q==0:
            g2f12 =  temp
        else:
            g2f12=np.hstack( [g2f12,  temp] ) 
    return g2f12



#####################################
#get one-time @different age
#####################################
 
def get_qedge( qstart,qend,qwidth,noqs,  ):
    ''' DOCUMENT make_qlist( )
    give qstart,qend,qwidth,noqs
    return a qedge by giving the noqs, qstart,qend,qwidth.
           a qcenter, which is center of each qedge 
    KEYWORD:  None    ''' 
    import numpy as np 
    qcenter = np.linspace(qstart,qend,noqs)
    #print ('the qcenter is:  %s'%qcenter )
    qedge=np.zeros(2*noqs) 
    qedge[::2]= (  qcenter- (qwidth/2)  ) #+1  #render  even value
    qedge[1::2]= ( qcenter+ qwidth/2) #render odd value
    return qedge, qcenter  


def rotate_g12q_to_rectangle( g12q ):
        
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Rotate anti clockwise 45 of a one-q two correlation function along diagonal to a masked array 
    the shape ( imgs_length, imgs_length )  of g12q will change to ( imgs_length, 2*imgs_length -1)

       
    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length ) 
  
    
    Return:
        g12qr: a masked 2-D array, shape as ( imgs_length, 2*imgs_length -1 )  
           x-axis: taus, from 0 to imgs_length
           y-axis: ages, from 0 to imgs_length( the middle of y) to 2imgs_length-1 (top) 
    One example:     
        g12qr = rotate_g12q_to_rectangle(g12bm[:,:,0] )
    '''
    M,N = g12q.shape
    g12qr = np.ma.empty(( 2*N-1,N ))
    g12qr.mask = True
    for i in range(N):
        g12qr[i:(2*N-1-i):2, i  ] = g12q.diagonal(i)
    return g12qr



def get_aged_g2_from_g12q( g12q, slice_num = 6, slice_width=5, slice_start=0, slice_end= 1  ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Get one-time correlation function of different age from two correlation function
    namely, calculate the different aged mean of each diag line of g12 to get one-time correlation fucntion
    
    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length ) 
    
    Options:
        slice_num: int, the slice number of the diagonal of g12
        slice_width: int,  each slice width in unit of pixel
        slice start: int, can start from 0
        slice end: int, can end at 2*imgs_length -1 
   
   
    Return:
        g2_aged: a dict, one time correlation function at different age
                 the keys of dict is ages in unit of pixel
                 dict[key]:    
                           a 1-D array, shape as ( imgs_length ), 
                           a one-q one-time correlation function
    One example:     
        g2_aged = get_aged_g2_from_g12q( g12q, slice_num =3, slice_width= 500, 
                slice_start=4000, slice_end= 20000-4000  )
    '''
 
    
    arr= rotate_g12q_to_rectangle( g12q )
    m,n = arr.shape #m should be 2*n-1
    age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
                     qwidth = slice_width, noqs =slice_num  )    
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)  
    #print (age_edge, age_center)
    g2_aged = {}
    for i,age in enumerate(age_center):         
        age_edges_0, age_edges_1 = age_edge[ i*2 : 2*i+2]  
        g2i = arr[ age_edges_0: age_edges_1   ].mean( axis =0 )
        g2i_ = np.array( g2i )       
        g2_aged[age] =   g2i_[np.nonzero( g2i_)[0]]
        
    return g2_aged 

def show_g12q_aged_g2( g12q, g2_aged,slice_width=10, timeperframe=1,vmin= 1, vmax= 1.25 ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Plot one-time correlation function of different age with two correlation function 
    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length ) 
        g2_aged: a dict, one time correlation function at different age
                 obtained by: for example,
                 g2_aged = get_aged_g2_from_g12q( g12q, slice_num =3, slice_width= 500, 
                                                  slice_start=4000, slice_end= 20000-4000  )
                 the keys of dict is ages in unit of pixel
                 dict[key]:    
                           a 1-D array, shape as ( imgs_length ), 
                           a one-q one-time correlation function                          
                           
    Options:
        slice_width: int,  each slice width in unit of pixel, for line width of a plot
        timeperframe: float, time per frame for axis unit
        vmin, float, matplot vmin
        vmax, float, matplot vmax
   
    Return:
         two plots, one for the two-time correlation, g12q, 
 
    One example:     
        show_g12q_aged_g2( g12q, g2_aged,timeperframe=1,vmin= 1, vmax= 1.22 )
    '''
    
    age_center = list( g2_aged.keys() )
    print ('the cut age centers are: ' +str(age_center)     )
    M,N = g12q.shape

    #fig, ax = plt.subplots( figsize = (8,8) )
    
    figw =10
    figh = 10
    fig = plt.figure(figsize=(figw,figh)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 8],height_ratios=[8,8]   ) 
    ax = plt.subplot(gs[0])     
    ax1 = plt.subplot(gs[1])     
    
    im=ax.imshow( g12q, origin='lower' , cmap='viridis', 
             norm= LogNorm( vmin, vmax ) , extent=[0, N, 0, N ] )

    linS = []
    linE=[]
    linS.append( zip( [0]*len(age_center), np.int_(age_center) ))
    linE.append( zip(  np.int_(age_center), [0]*len(age_center) ))
    for i, [ps,pe] in enumerate(zip(linS[0],linE[0])):     
        if ps[1]>=N:s0=ps[1] - N;s1=N
        else:s0=0;s1=ps[1]        
        if pe[0]>=N:e0=N;e1=pe[0] - N
        else:e0=pe[0];e1=0     
        lined= slice_width/2.  #in data width
        linewidth=    (lined * (figh*72./N)) * 0.8
        ax.plot( [s0,e0],[s1,e1], linewidth=linewidth ,alpha=0.3 )  #, color=   )  
    
    ax.set_title(  '%s_frames'%(N)    )
    ax.set_xlabel( r'$t_1$ $(s)$', fontsize = 18)
    ax.set_ylabel( r'$t_2$ $(s)$', fontsize = 18)
    fig.colorbar(im) 
    
    ax1.set_title("Aged_G2")
    for i in sorted(g2_aged.keys()):
        gx= np.arange(len(g2_aged[i])) * timeperframe
        marker = next(markers)        
        ax1.plot( gx,g2_aged[i], '-%s'%marker, label=r"$age= %.1f s$"%(i*timeperframe))
        ax1.set_ylim( vmin, vmax )
        ax1.set_xlabel(r"$\tau $ $(s)$", fontsize=18) 
        ax1.set_ylabel("g2")
        ax1.set_xscale('log')
    ax1.legend(fontsize='small', loc='best' ) 
    #plt.show()
    


 

#####################################
#get fout-time

def get_tau_from_g12q( g12q, slice_num = 6, slice_width=1, slice_start=None, slice_end=None  ):
    
        
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Get tau lines from two correlation function
    namely, get diag line of g12 as a function of ages
    
    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length ) 
    
    Options:
        slice_num: int, the slice number of the diagonal of g12
        slice_width: int,  each slice width in unit of pixel
        slice start: int, can start from 0
        slice end: int, can end at  imgs_length -1 
   
   
    Return:
        return: tau, a dict, tau lines  
                 the keys of dict is tau(slice center) in unit of pixel
                 dict[key]:    
                           a 1-D array, shape as ( tau_line-length ), 
                            
    One example:     
        taus = get_aged_g2_from_g12q( g12q, slice_num =3, slice_width= 500, 
                slice_start=4000, slice_end= 20000-4000  )
    '''
    
    
  
    arr= rotate_g12q_to_rectangle( g12q )
    m,n = arr.shape #m should be 2*n-1
 
    age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
                     qwidth = slice_width, noqs =slice_num  )    
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)  
    #print (age_edge, age_center)
    tau = {}
    for i,age in enumerate(age_center):         
        age_edges_0, age_edges_1 = age_edge[ i*2 : 2*i+2] 
        #print (age_edges_0, age_edges_1)
        g2i = arr[ :,age_edges_0: age_edges_1   ].mean( axis =1 )
        g2i_ = np.array( g2i )       
        tau[age] =   g2i_[np.nonzero( g2i_)[0]]
        
    return tau




def show_g12q_taus( g12q, taus,  slice_width=10, timeperframe=1,vmin= 1, vmax= 1.25 ):    
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Plot tau-lines as a function of age with two correlation function 
 
    
    Parameters:
        g12q: a 2-D array, one-q two correlation function, shape as ( imgs_length, imgs_length )
        tau, a dict, tau lines  
                 the keys of dict is tau(slice center) in unit of pixel
                 dict[key]:    
                           a 1-D array, shape as ( tau_line-length ), 
                 obtained by: for example,
                         taus = get_tau_from_g12q( g12b_norm[:,:,0], slice_num = 5, slice_width=1, 
                  slice_start=3, slice_end= 5000-1  ))
                         
                           
    Options:
        slice_width: int,  each slice width in unit of pixel, for line width of a plot
        timeperframe: float, time per frame for axis unit
        vmin, float, matplot vmin
        vmax, float, matplot vmax
   
    Return:
         two plots, one for tau lines~ages, g12q, 
 
    One example:     
        show_g12q_taus( g12b_norm[:,:,0], taus,  slice_width=50, 
               timeperframe=1,vmin=1.01,vmax=1.55 )
    '''
     
     
     
    age_center = list( taus.keys() )
    print ('the cut tau centers are: ' +str(age_center)     )
    M,N = g12q.shape

    #fig, ax = plt.subplots( figsize = (8,8) )
    
    figw =10
    figh = 10
    fig = plt.figure(figsize=(figw,figh)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 8],height_ratios=[8,8]   ) 
    ax = plt.subplot(gs[0])     
    ax1 = plt.subplot(gs[1])     
    
    im=ax.imshow( g12q, origin='lower' , cmap='viridis', 
             norm= LogNorm( vmin= vmin, vmax= vmax ) , extent=[0, N, 0, N ] )

    linS = []
    linE=[]
    linS.append( zip(  np.int_(age_center) -1, [0]*len(age_center)   ))
    linE.append( zip(  [N -1]*len(age_center), N  - np.int_(age_center)   ))
    for i, [ps,pe] in enumerate(zip(linS[0],linE[0])):
        lined= slice_width  #/2. *draw_scale_tau  #in data width
        linewidth=    (lined * (figh*72./N)) * 0.8
        #print (ps,pe)
        ax.plot( [ps[0],pe[0]],[ps[1],pe[1]], linewidth=linewidth ) #, color=   )  
    
    ax.set_title(  '%s_frames'%(N)    )
    ax.set_xlabel( r'$t_1$ $(s)$', fontsize = 18)
    ax.set_ylabel( r'$t_2$ $(s)$', fontsize = 18)
    fig.colorbar(im)    
    
    ax1.set_title("Tau_Cuts_in_G12")
    for i in sorted(taus.keys()):
        gx= np.arange(len(taus[i])) * timeperframe
        marker = next(markers)        
        ax1.plot( gx,taus[i], '-%s'%marker, label=r"$tau= %.1f s$"%(i*timeperframe))
        ax1.set_ylim( vmin,vmax )
        ax1.set_xlabel(r'$t (s)$',fontsize=5)
        ax1.set_ylabel("g2")
        ax1.set_xscale('log')
    ax1.legend(fontsize='small', loc='best' ) 
    #plt.show()
    
 


def histogram_taus(taus, hisbin=20, plot=True,timeperframe=1):
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Do histogram and plot of tau-lines  
 
    
    Parameters:
       taus, a dict, tau lines  
                 the keys of dict is tau(slice center) in unit of pixel
                 dict[key]:    
                           a 1-D array, shape as ( tau_line-length ), 
                 obtained by: for example,
                         taus = get_tau_from_g12q( g12b_norm[:,:,0], slice_num = 5, slice_width=1, 
                  slice_start=3, slice_end= 5000-1  )) 
                           
    Options:
        bins: int,  bins number for the histogram
        plot: if True, show the histogram plot
        timeperframe: float, time per frame for axis unit
 
   
    Return:
         his:  a dict, his[key], the histogram of tau-lines
         if plot, plot the histogram  of tau-lines  
 
    One example:     
        his = histogram_taus(taus, hisbin=30, plot=True, timeperframe=timeperframe)
    '''    
    
    
    his={}
    for key in list(taus.keys()):
        his[key] = np.histogram( taus[key], bins=hisbin)
        
    if plot:            
        fig, ax1 = plt.subplots(figsize=(8, 8))        
        ax1.set_title("Tau_histgram")
        for key in sorted(his.keys()):
            tx= 0.5*( his[key][1][:-1] + his[key][1][1:])
            marker = next(markers)       
            ax1.plot( tx, his[key][0], '-%s'%marker, label=r"$tau= %.1f s$"%(key*timeperframe) )
            #ax1.set_ylim( 1.05,1.35 )
            ax1.set_xlim( 1.05,1.35 )
            ax1.set_xlabel(r'$g_2$',fontsize=19)
            ax1.set_ylabel(r"histgram of g2 @ tau",fontsize=15)
            #ax1.set_xscale('log')
        ax1.legend(fontsize='large', loc='best' ) 
        #plt.show()
        
    return his
        




    
    


def get_four_time_from_two_time(  g12,g2=None, rois=None  ):
    ''' 
    Dec 16, 2015, Y.G.@CHX
    Get four-time correlation function from two correlation function
    namely, calculate the deviation of each diag line of g12 to get four-time correlation fucntion
    TOBEDONE: deal with bad frames
    
    Parameters:
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q)  
    
    Options:
        g2: if not None, a 2-D array, shape as ( imgs_length,  q), or (tau, q)
            one-time correlation fucntion, for normalization of the four-time
        rois: if not None, a list, [x-slice-start, x-slice-end, y-slice-start, y-slice-end]
   
    Return:
        g4f12: a 2-D array, shape as ( imgs_length,  q), 
                   a four-time correlation function  
     
    One example:        
        s1,s2 = 0,2000
        g4 = get_four_time_from_two_time( g12bm, g2b, roi=[s1,s2,s1,s2] )
         
    '''      
    
    
    m,n,noqs = g12.shape
    g4f12 = []       
    for q in  range(noqs):   
        temp=[]    
        if rois is None:
            y=g12[:,:,q]  
        else:
            x1,x2,y1,y2 = rois
            y=g12[x1:x2,y1:y2, q]
            m,n = y.shape
        norm =  ( g2[:,q][0] -1)**2  
        for tau in range(m): 
            d_ = np.diag(y,k=int(tau))
            d = d_[   np.where( d_ !=1)            ]
            g4 = ( d.std() )**2 /norm
            temp.append( g4 )                
                
        temp = np.array( temp).reshape( len(temp),1)
        if q==0:
            g4f12 =  temp
        else:
            g4f12=np.hstack( [g4f12,  temp] ) 
            
    return g4f12








######
def make_g12_mask(  badframes_list, g12_shape):
    ''' 
    Dec 16, 2015, Y.G.@CHX
    make g12 mask by badlines
    
    Parameters: 
        badframes_list: list, contains the bad frame number, like [100, 155, 10000]  
        g12_shape: the shape of one-q two correlation function, shape as ( imgs_length, imgs_length )
    Return:
        g12_mask: a 2-D array, shape as ( imgs_length, imgs_length )
                    
     
    One example:        
        g12_mask = make_g12_mask(bad_frames, g12b[:,:,0].shape)         
           
    '''
    
    m,n  = g12_shape
    #g12_mask = np.ma.empty( ( m,n ) )
    g12_mask = np.ma.ones( ( m,n ) )
    g12_mask.mask= False    
    for bdl in badframes_list:
        g12_mask.mask[:,bdl] = True
        g12_mask.mask[bdl,:] = True
    return g12_mask



def masked_g12( g12, badframes_list):
    ''' 
    Dec 16, 2015, Y.G.@CHX
    make masked g12 with mask defined by badframes_list
     
    
    Parameters:  
        g12: a 3-D array, two correlation function, shape as ( imgs_length, imgs_length, q) 
        badframes_list: list, contains the bad frame number, like [100, 155, 10000]     
 
    Return:
        g12m: a masked 3-D array, shape as same as g12, ( imgs_length, imgs_length, q )
                    
     
    One example:        
        g12m  =  masked_g12( g12b, bad_frames) 
 
    '''
    
    m,n,qs = g12.shape
    g12m = np.ma.empty_like( g12 )
    g12_mask = make_g12_mask(  badframes_list, g12[:,:,0].shape) 
 
    for i in range(qs):
        g12m[:,:,i] = g12[:,:,i] * g12_mask
    return g12m




def make_norms_mask(norms, badlines):
    '''make g12 maks to mask teh badlines'''
    
    m,qs = norms.shape     
    norms_mask = np.ma.ones( ( m ) )
    norms_mask.mask= False    
    for bdl in badlines:
        norms_mask.mask[ bdl] = True
         
    return norms_mask


def masked_norms( norms, norms_mask):
    '''mask g12 by badlines'''
    
    m,qs = norms.shape
    norms_ = np.ma.empty_like( norms )
    #norms_mask = make_norms_mask(norms, badlines)
    for i in range(qs):
        norms_[:,i] = norms[:,i] * norms_mask
    return norms_








        
    
def delays( num_lev=3, num_buf=4, time=1 ): 
    ''' DOCUMENT delays(time=)
        return array of delays.
        KEYWORD:  time: scale delays by time ( should be time between frames)
     '''
    if num_buf%2!=0:print ("nobuf must be even!!!"    )
    dly=np.zeros( (num_lev+1)*int(num_buf/2) +1  )        
    dict_dly ={}
    for i in range( 1,num_lev+1):
        if i==1:imin= 1
        else:imin= int(num_buf/2)+1
        ptr=(i-1)*int(num_buf/2)+ np.arange(imin,num_buf+1)
        dly[ptr]= np.arange( imin, num_buf+1) *2**(i-1)            
        dict_dly[i] = dly[ptr-1]            
        dly*=time
        #print (i, ptr, imin)
    return dly, dict_dly
            
     
    

class Get_Pixel_Array(object):
    '''
    a class to get intested pixels from a images sequence, 
    load ROI of all images into memory 
    get_data: to get a 2-D array, shape as (len(images), len(pixellist))
    
    One example:        
        data_pixel =   Get_Pixel_Array( imgsr, pixelist).get_data()
    '''
    
    def __init__(self, indexable, pixelist):
        '''
        indexable: a images sequences
        pixelist:  1-D array, interest pixel list
        '''
        self.indexable = indexable
        self.pixelist = pixelist
        #self.shape = indexable.shape
        try:
            self.length= len(indexable)
        except:
            self.length= indexable.length           
            
    def get_data(self ): 
        '''
        To get intested pixels array
        Return: 2-D array, shape as (len(images), len(pixellist))
        '''
        
        #print (self.length)
        data_array = np.zeros([ self.length,len(self.pixelist)])
        for key in range(self.length ):
            data_array[key] = np.ravel( self.indexable[key])[self.pixelist]  
        return data_array
     
    
 


class Reverse_Coordinate(object):
    def __init__(self, indexable, mask):
        self.indexable = indexable
        self.mask = mask
        try:
            self.shape  = indexable.shape
        except:
            #if 
            self.shape  =  [len(indexable), indexable[0].shape[0], indexable[0].shape[1] ]
        #self.shape = indexable.shape
        self.length= len(indexable)
    def __getitem__(self, key ):      
        if self.mask is not None:
            img =self.indexable[key] * self.mask  
        else:
            img = self.indexable[key]
            
        if len(img.shape) ==3:
            img_=img[:,::-1,:]
        if len(img.shape)==2:
            img_=img[::-1,:] 
        return img_ 

 

def get_mean_intensity( data_pixel, qind):
    ''' 
    Dec 16, 2015, Y.G.@CHX
    a function to get mean intensity as a function of time (image number)
    
    Parameters:
        data_pixel: 2-D array, shape as (len(images), len(qind)),
                    use function Get_Pixel_Array( ).get_data(  ) to get
        qind: 1-D int array, a index list of interest pixel, values change from 1 to int number
   
    Return:
        mean_inten: a dict, with keys as the unique values of qind, 
                    each dict[key]: 1-D array, with shape as data_pixel.shape[0],namely, len(images)
     
    One example:        
        mean_inten = get_mean_intensity( data_pixel, qind)
    '''
    
    noqs = len( np.unique(qind) )
    mean_inten = {}
               
    for qi in range(1, noqs + 1 ):
        pixelist_qi =  np.where( qind == qi)[0] 
        #print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi =    data_pixel[:,pixelist_qi]  
        mean_inten[qi] =  data_pixel_qi.mean( axis =1 )
    return  mean_inten




def show_C12(C12, qz_ind=0, qr_ind=0, N1=None,N2=None, vmin=None, vmax=None, title=False):
    
    g12_num =  qz_ind * num_qr + qr_ind
    if N1 is None:
        N1=0
    if N2 is None:
        N2=Nming
    if vmin is None:
        vmin = 1
    if vmax is None:
        vmax = 1.02
    data = g12b[N1:N2,N1:N2,g12_num]
    fig, ax = plt.subplots()
    im=ax.imshow( data, origin='lower' , cmap='viridis', 
                 norm= LogNorm( vmin, vmax ), 
            extent=[0, data.shape[0]*timeperframe, 0, data.shape[0]*timeperframe ] )
    #ax.set_title('%s-%s frames--Qth= %s'%(N1,N2,g12_num))
    if title:
        ax.set_title('%s-%s frames--Qz= %s--Qr= %s'%(N1,N2, qz_center[qz_ind], qr_center[qr_ind] ))
    ax.set_xlabel( r'$t_1$ $(s)$', fontsize = 18)
    ax.set_ylabel( r'$t_2$ $(s)$', fontsize = 18)
    fig.colorbar(im)
    
    #plt.show()    


def show_C12(C12, q_ind=0,  *argv,**kwargs):  
 
    '''
    plot one-q of two-time correlation function
    C12: two-time correlation function, with shape as [ time, time, qs]
    q_ind: if integer, for a SAXS q, the nth of q to be plotted
            if a list: for a GiSAXS [qz_ind, qr_ind]  
    kwargs: support        
        timeperframe: the time interval
        N1: the start frame(time)
        N2: the end frame(time)
        vmin/vmax: for plot
        title: if True, show the tile
    
    e.g.,
        show_C12(g12b, q_ind=1, N1=0, N2=500, vmin=1.05, vmax=1.07,  )
    
    '''
  
    #strs =  [ 'timeperframe', 'N1', 'N2', 'vmin', 'vmax', 'title'] 
    
    shape = C12.shape
    if isinstance(q_ind, int):
        C12_num = q_ind
    else:
        qz_ind, qr_ind = q_ind
        C12_num =  qz_ind * num_qr + qr_ind 
    
    if 'timeperframe' in kwargs.keys():
        timeperframe =  kwargs['timeperframe']
    else:
        timeperframe=1
        
    if 'vmin' in kwargs.keys():
        vmin =  kwargs['vmin']
    else:
        vmin=1
    if 'vmax' in kwargs.keys():
        vmax =  kwargs['vmax']
    else:
        vmax=1.05        
        
    if 'N1' in kwargs.keys():
        N1 =  kwargs['N1']
    else:
        N1=0
        
    if 'N2' in kwargs.keys():
        N2 =  kwargs['N2']
    else:
        N2= shape[0]
    if 'title' in kwargs.keys():
        title =  kwargs['title']
    else:
        title=True        

    data = C12[N1:N2,N1:N2,C12_num]
    fig, ax = plt.subplots()
    im=ax.imshow( data, origin='lower' , cmap='viridis', 
                 norm= LogNorm( vmin, vmax ), 
            extent=[0, data.shape[0]*timeperframe, 0, data.shape[0]*timeperframe ] )
    if title:
        if isinstance(q_ind, int):
            ax.set_title('%s-%s frames--Qth= %s'%(N1,N2,C12_num))
        else:
            ax.set_title('%s-%s frames--Qzth= %s--Qrth= %s'%(N1,N2, qz_ind, qr_ind ))
        
        #ax.set_title('%s-%s frames--Qth= %s'%(N1,N2,g12_num))
    ax.set_xlabel( r'$t_1$ $(s)$', fontsize = 18)
    ax.set_ylabel( r'$t_2$ $(s)$', fontsize = 18)
    fig.colorbar(im)
    #plt.show()   




def plot_saxs_g2( g2, taus,  *argv,**kwargs):     
    '''plot g2 results, 
       g2: one-time correlation function
       taus: the time delays        
       kwargs: can contains
           vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
           ylim/xlim: the limit of y and x
       
       e.g.
       plot_saxs_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )
           
      ''' 
     
    
    if 'uid' in kwargs.keys():
        uid = kwargs['uid'] 
    else:
        uid = 'uid'
        
    if 'q_ring_center' in kwargs.keys():
        q_ring_center = kwargs[ 'q_ring_center']
    else:
        q_ring_center = np.arange(  g2.shape[1] )
    

    
    num_rings = g2.shape[1]
    sx = int(round(np.sqrt(num_rings)) )
    if num_rings%sx == 0: 
        sy = int(num_rings/sx)
    else:
        sy=int(num_rings/sx+1)
        
    uid = 'uid'
    if 'uid' in kwargs.keys():
        uid = kwargs['uid']  
        
    fig = plt.figure(figsize=(14, 10))
    plt.title('uid= %s'%uid,fontsize=20, y =1.06)
    plt.axis('off')
    #plt.axes(frameon=False)
    plt.xticks([])
    plt.yticks([])
    for i in range(num_rings):
        ax = fig.add_subplot(sx, sy, i+1 )
        ax.set_ylabel("g2") 
        ax.set_title(" Q= " + '%.5f  '%(q_ring_center[i]) + r'$\AA^{-1}$')
        y=g2[:, i]
        ax.semilogx(taus, y, '-o', markersize=6) 
        #ax.set_ylim([min(y)*.95, max(y[1:])*1.05 ])
        if 'ylim' in kwargs:
            ax.set_ylim( kwargs['ylim'])
        elif 'vlim' in kwargs:
            vmin, vmax =kwargs['vlim']
            ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
        else:
            pass
        if 'xlim' in kwargs:
            ax.set_xlim( kwargs['xlim'])
 
    #plt.show()
    fig.tight_layout() 
    
    
