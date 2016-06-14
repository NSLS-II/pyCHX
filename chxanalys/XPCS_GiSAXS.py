from .chx_generic_functions import *



##########################################
###Functions for GiSAXS
##########################################


def make_gisaxs_grid( qr_w= 10, qz_w = 12, dim_r =100,dim_z=120):
    y, x = np.indices( [dim_z,dim_r] )
    Nr = int(dim_r/qp_w)
    Nz = int(dim_z/qz_w)
    noqs = Nr*Nz
    
    ind = 1
    for i in range(0,Nr):
        for j in range(0,Nz):        
            y[ qr_w*i: qr_w*(i+1), qz_w*j:qz_w*(j+1)]=  ind
            ind += 1 
    return y 


###########################################
#for Q-map, convert pixel to Q
########################################### 


def get_incident_angles( inc_x0, inc_y0, refl_x0, refl_y0, pixelsize=[75,75], Lsd=5.0):
    ''' giving: incident beam center: bcenx,bceny
                reflected beam on detector: rcenx, rceny
                sample to detector distance: Lsd, in meters
                pixelsize: 75 um for Eiger4M detector
        get incident_angle (alphai), the title angle (phi)
    '''
    px,py = pixelsize
    phi = np.arctan2( (refl_x0 - inc_x0)*px *10**(-6), (refl_y0 - inc_y0)*py *10**(-6) )    
    alphai = np.arctan2( (refl_y0 -inc_y0)*py *10**(-6),  Lsd ) /2.     
    #thetai = np.arctan2(  (rcenx - bcenx)*px *10**(-6), Lsd   ) /2.  #??   
    
    return alphai,phi 
    
       
def get_reflected_angles(inc_x0, inc_y0, refl_x0, refl_y0, thetai=0.0,
                         pixelsize=[75,75], Lsd=5.0,dimx = 2070.,dimy=2167.):
    
    ''' giving: incident beam center: bcenx,bceny
                reflected beam on detector: rcenx, rceny
                sample to detector distance: Lsd, in meters                
                pixelsize: 75 um for Eiger4M detector
                detector image size: dimx = 2070,dimy=2167 for Eiger4M detector
        get  reflected angle alphaf (outplane)
             reflected angle thetaf (inplane )
    '''    
    
    alphai, phi =  get_incident_angles( inc_x0, inc_y0, refl_x0, refl_y0, pixelsize, Lsd)
    print ('The incident_angle (alphai) is: %s'%(alphai* 180/np.pi))
    px,py = pixelsize
    y, x = np.indices( [dimy,dimx] )    
    #alphaf = np.arctan2( (y-inc_y0)*py*10**(-6), Lsd )/2 - alphai 
    alphaf = np.arctan2( (y-inc_y0)*py*10**(-6), Lsd )  - alphai 
    thetaf = np.arctan2( (x-inc_x0)*px*10**(-6), Lsd )/2 - thetai   
    
    return alphaf,thetaf, alphai, phi    
    
    

def convert_gisaxs_pixel_to_q( inc_x0, inc_y0, refl_x0, refl_y0, 
                               pixelsize=[75,75], Lsd=5.0,dimx = 2070.,dimy=2167.,
                              thetai=0.0, lamda=1.0 ):
    
    ''' 
    
    giving: incident beam center: bcenx,bceny
                reflected beam on detector: rcenx, rceny
                sample to detector distance: Lsd, in meters                
                pixelsize: 75 um for Eiger4M detector
                detector image size: dimx = 2070,dimy=2167 for Eiger4M detector                
                wavelength: angstron               
                
        get: q_parallel (qp), q_direction_z (qz)
                
    '''         
    
    
    alphaf,thetaf,alphai, phi = get_reflected_angles( inc_x0, inc_y0, refl_x0, refl_y0, thetai, pixelsize, Lsd,dimx,dimy)
       
    pref = 2*np.pi/lamda
    
    qx = np.cos( alphaf)*np.cos( 2*thetaf) - np.cos( alphai )*np.cos( 2*thetai)  
    qy_ = np.cos( alphaf)*np.sin( 2*thetaf) - np.cos( alphai )*np.sin ( 2*thetai)    
    qz_ = np.sin(alphaf) + np.sin(alphai) 
    
    
    qy = qz_* np.sin( phi) + qy_*np.cos(phi) 
    qz = qz_* np.cos( phi) - qy_*np.sin(phi)   
    
    qr = np.sqrt( qx**2 + qy**2 ) 
    
    
    return qx*pref  , qy*pref  , qr*pref  , qz*pref  
        
    
    
    
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
    
    
###########################################
#for plot Q-map 
###########################################     
    
def get_qmap_label( qmap, qedge ):
    import numpy as np
    '''give a qmap and qedge to bin the qmap into a label array'''
    edges = np.atleast_2d(np.asarray(qedge)).ravel()
    label_array = np.digitize(qmap.ravel(), edges, right=False)
    label_array = np.int_(label_array)
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    label_array = label_array.reshape( qmap.shape )
    return label_array
        
    
    
def get_qzrmap(label_array_qz, label_array_qr, qz_center, qr_center   ):
    '''get   qzrmap  '''
    qzmax = label_array_qz.max()
    label_array_qr_ = np.zeros( label_array_qr.shape  )
    ind = np.where(label_array_qr!=0)
    label_array_qr_[ind ] =  label_array_qr[ ind ] + 1E4  #add some large number to qr
    label_array_qzr = label_array_qz * label_array_qr_  
    
    #convert label_array_qzr to [1,2,3,...]
    uqzr = np.unique( label_array_qzr )[1:]
    
    uqz = np.unique( label_array_qz )[1:]
    uqr = np.unique( label_array_qr )[1:]
    #print (uqzr)
    label_array_qzr_ = np.zeros_like( label_array_qzr )
    newl = np.arange( 1, len(uqzr)+1)
    
    qzc =list(qz_center) * len( uqr )
    qrc= [  [qr_center[i]]*len( uqz ) for i in range(len( uqr ))  ]
    
    for i, label in enumerate(uqzr):
        #print (i, label)
        label_array_qzr_.ravel()[ np.where(  label_array_qzr.ravel() == label)[0] ] = newl[i]    
    
    
    return np.int_(label_array_qzr_), np.array( qzc ), np.concatenate(np.array(qrc ))
    


def show_label_array_on_image(ax, image, label_array, cmap=None,norm=None, log_img=True,alpha=0.3,
                              imshow_cmap='gray', **kwargs):  #norm=LogNorm(), 
    """
    This will plot the required ROI's(labeled array) on the image
    Additional kwargs are passed through to `ax.imshow`.
    If `vmin` is in kwargs, it is clipped to minimum of 0.5.
    Parameters
    ----------
    ax : Axes
        The `Axes` object to add the artist too
    image : array
        The image array
    label_array : array
        Expected to be an unsigned integer array.  0 is background,
        positive integers label region of interest
    cmap : str or colormap, optional
        Color map to use for plotting the label_array, defaults to 'None'
    imshow_cmap : str or colormap, optional
        Color map to use for plotting the image, defaults to 'gray'
    norm : str, optional
        Normalize scale data, defaults to 'Lognorm()'
    Returns
    -------
    im : AxesImage
        The artist added to the axes
    im_label : AxesImage
        The artist added to the axes
    """
    ax.set_aspect('equal')
    if log_img:
        im = ax.imshow(image, cmap=imshow_cmap, interpolation='none',norm=LogNorm(norm),**kwargs)  #norm=norm,
    else:
        im = ax.imshow(image, cmap=imshow_cmap, interpolation='none',norm=norm,**kwargs)  #norm=norm,
        
    im_label = mpl_plot.show_label_array(ax, label_array, cmap=cmap, norm=norm, alpha=alpha,
                                **kwargs)  # norm=norm,
    
    
    return im, im_label    
    
    
 


def show_qz(qz):
    ''' 
    plot qz mape

    '''
        
        
    fig, ax = plt.subplots()
    im=ax.imshow(qz, origin='lower' ,cmap='viridis',vmin=qz.min(),vmax= qz.max() )
    fig.colorbar(im)
    ax.set_title( 'Q-z')
    plt.show()
    
def show_qr(qr):
    ''' 
    plot qr mape

    '''
    fig, ax = plt.subplots()    
    im=ax.imshow(qr, origin='lower' ,cmap='viridis',vmin=qr.min(),vmax= qr.max() )
    fig.colorbar(im)
    ax.set_title( 'Q-r')
    plt.show()    

def show_alphaf(alphaf,):
    ''' 
     plot alphaf mape

    '''
        
    fig, ax = plt.subplots()
    im=ax.imshow(alphaf*180/np.pi, origin='lower' ,cmap='viridis',vmin=-1,vmax= 1.5 )
    #im=ax.imshow(alphaf, origin='lower' ,cmap='viridis',norm= LogNorm(vmin=0.0001,vmax=2.00))    
    fig.colorbar(im)
    ax.set_title( 'alphaf')
    plt.show()    
    



    
########################
# get one-d of I(q) as a function of qr for different qz
##################### 
    

    
    
def get_1d_qr(  data, Qr,Qz, qr, qz, inc_x0,  mask=None, show_roi=True,  ticks=None, alpha=0.3 ): 
    '''
       plot one-d of I(q) as a function of qr for different qz
       data: a dataframe
       Qr: info for qr, = qr_start , qr_end, qr_width, qr_num
       Qz: info for qz, = qz_start,   qz_end,  qz_width , qz_num
       qr: qr-map
       qz: qz-map
       inc_x0: x-center of incident beam
       mask: a mask for qr-1d integration
       show_roi: boolean, if ture, show the interest ROI
       ticks: ticks for the plot, = zticks, zticks_label, rticks, rticks_label
       alpha: transparency of ROI
       Return: qr_1d, a dict, with keys as qz number
                      qr_1d[key]: 
               Plot 1D cureve as a function of Qr for each Qz  
               
    Examples:
        #to make two-qz, from 0.018 to 0.046, width as 0.008,
        qz_width = 0.008
        qz_start = 0.018 + qz_width/2
        qz_end = 0.046  -  qz_width/2
        qz_num= 2


        #to make one-qr, from 0.02 to 0.1, and the width is 0.1-0.012
        qr_width =  0.1-0.02
        qr_start =    0.02 + qr_width  /2
        qr_end =  0.01 -  qr_width  /2
        qr_num = 1

        Qr = [qr_start , qr_end, qr_width, qr_num]
        Qz=  [qz_start,   qz_end,  qz_width , qz_num ]
        new_mask[ :, 1020:1045] =0
        ticks = show_qzr_map(  qr,qz, inc_x0, data = avg_imgmr, Nzline=10,  Nrline=10   )
        qx, qy, qr, qz = convert_gisaxs_pixel_to_q( inc_x0, inc_y0,refl_x0,refl_y0, lamda=lamda, Lsd=Lsd )
        
        qr_1d = get_1d_qr( avg_imgr, Qr, Qz, qr, qz, inc_x0,  new_mask,  True, ticks, .8)


    '''               
 
    
    qr_start , qr_end, qr_width, qr_num =Qr
    qz_start,   qz_end,  qz_width , qz_num =Qz
    qr_edge, qr_center = get_qedge(qr_start , qr_end, qr_width, qr_num )    
    qz_edge, qz_center = get_qedge( qz_start,   qz_end,  qz_width , qz_num ) 
     
    print ('The qr_edge is:  %s\nThe qr_center is:  %s'%(qr_edge, qr_center))
    print ('The qz_edge is:  %s\nThe qz_center is:  %s'%(qz_edge, qz_center))    
    label_array_qr = get_qmap_label( qr, qr_edge)

    if show_roi:
        label_array_qz0 = get_qmap_label( qz , qz_edge)
        label_array_qzr0,qzc0,qrc0 = get_qzrmap(label_array_qz0, label_array_qr,qz_center, qr_center  )  
        
        if mask is not None:label_array_qzr0 *= mask
        #data_ = data*label_array_qzr0           
        show_qzr_roi( data,label_array_qzr0, inc_x0, ticks, alpha)

    fig, ax = plt.subplots()
    qr_1d ={}
    for i,qzc_ in enumerate(qz_center):
        
        #print (i,qzc_)
        label_array_qz = get_qmap_label( qz, qz_edge[i*2:2*i+2])
        #print (qzc_, qz_edge[i*2:2*i+2])
        label_array_qzr,qzc,qrc = get_qzrmap(label_array_qz, label_array_qr,qz_center, qr_center  )
        #print (np.unique(label_array_qzr ))    
        if mask is not None:label_array_qzr *=   mask
        roi_pixel_num = np.sum( label_array_qzr, axis=0)
        qr_ = qr  *label_array_qzr
        data_ = data*label_array_qzr    
        qr_ave = np.sum( qr_, axis=0)/roi_pixel_num
        data_ave = np.sum( data_, axis=0)/roi_pixel_num     
        qr_1d[i]= [qr_ave, data_ave]
        ax.plot( qr_ave, data_ave,  '--o', label= 'qz= %f'%qzc_)
        
        
    #ax.set_xlabel( r'$q_r$', fontsize=15)
    ax.set_xlabel('$q $'r'($\AA^{-1}$)', fontsize=18)
    ax.set_ylabel('$Intensity (a.u.)$', fontsize=18)
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(   qr.max(),qr.min()  )
    ax.legend(loc='best')
    return qr_1d
   
def interp_zeros(  data ): 
    from scipy.interpolate import interp1d
    gf = data.ravel() 
    indice, = gf.nonzero() 
    start, stop = indice[0], indice[-1]+1 
    dx,dy = data.shape 
    x=np.arange( dx*dy ) 
    f = interp1d(x[indice], gf[indice]) 
    gf[start:stop] = f(x[start:stop]) 
    return gf.reshape([dx,dy])     


def get_qr_tick_label( qr, label_array_qr, inc_x0, interp=True):
    ''' 
    Dec 16, 2015, Y.G.@CHX
    get zticks,zticks_label 
    
    Parameters:
         
        qr:  2-D array, qr of a gisaxs image (data)
        label_array_qr: a labelled array of qr map, get by:
                        label_array_qr = get_qmap_label( qr, qz_edge)
    Options:
        interp: if True, make qz label round by np.round(data, 2)                
        inc_x0: x-center of incident beam
    Return:
        rticks: list, r-tick positions in unit of pixel
        rticks_label: list, r-tick positions in unit of real space   
    
    Examples:
        rticks,rticks_label = get_qr_tick_label( qr, label_array_qr)

    '''
        
    rticks =[]
    rticks_label = []
    num =  len( np.unique( label_array_qr ) )
    for i in range( 1, num   ):
        ind =  np.where( label_array_qr==i )[1]
        #tick = round( qr[label_array_qr==i].mean(),2)
        tick =  qr[label_array_qr==i].mean()
        if ind[0] < inc_x0 and ind[-1]>inc_x0:
             
            #mean1 = int( (ind[np.where(ind < inc_x0)[0]]).mean() )
            #mean2 = int( (ind[np.where(ind > inc_x0)[0]]).mean() )
            
            mean1 = int( (ind[np.where(ind < inc_x0)[0]])[0] )
            mean2 = int( (ind[np.where(ind > inc_x0)[0]])[0] )            
            
            
            
            rticks.append( mean1)
            rticks.append(mean2)
             
            rticks_label.append( tick )
            rticks_label.append( tick )
             
        else: 
            
            #print('here')
            #mean = int( ind.mean() )
            mean = int( ind[0] )
            rticks.append(mean)
            rticks_label.append( tick )
            #print (rticks)
            #print (mean, tick)
    
    if interp: 
        rticks =  np.array(rticks)
        rticks_label = np.array( rticks_label)        
        w= np.where( rticks <= inc_x0)[0] 
        rticks1 = np.int_(np.interp( np.round( rticks_label[w], 3), rticks_label[w], rticks[w]    )) 
        rticks_label1 = np.round( rticks_label[w], 3)  
        
        w= np.where( rticks > inc_x0)[0] 
        rticks2 = np.int_(np.interp( np.round( rticks_label[w], 3), rticks_label[w], rticks[w]    ))       
        rticks = np.append( rticks1, rticks2)   
        rticks_label2 = np.round( rticks_label[w], 3)  
        
        rticks_label = np.append( rticks_label1, rticks_label2)    
        
    return rticks, rticks_label


def get_qz_tick_label( qz, label_array_qz,interp=True):  
    ''' 
    Dec 16, 2015, Y.G.@CHX
    get zticks,zticks_label 
    
    Parameters:
         
        qz:  2-D array, qz of a gisaxs image (data)
        label_array_qz: a labelled array of qz map, get by:
                        label_array_qz = get_qmap_label( qz, qz_edge)
        interp: if True, make qz label round by np.round(data, 2)
 
    Return:
        zticks: list, z-tick positions in unit of pixel
        zticks_label: list, z-tick positions in unit of real space   
    
    Examples:
        zticks,zticks_label = get_qz_tick_label( qz, label_array_qz)

    '''
        
    num =  len( np.unique( label_array_qz ) )
    #zticks = np.array( [ int( np.where( label_array_qz==i )[0].mean() ) for i in range( 1,num ) ])
    zticks = np.array( [ int( np.where( label_array_qz==i )[0][0] ) for i in range( 1,num ) ])
    
    
    
    #zticks_label = np.array( [ round( qz[label_array_qz==i].mean(),4) for i in range( 1, num ) ])
    #zticks_label = np.array( [  qz[label_array_qz==i].mean() for i in range( 1, num ) ])
    zticks_label = np.array( [  qz[label_array_qz==i][0] for i in range( 1, num ) ])
    
    
    if interp:        
        zticks = np.int_(np.interp( np.round( zticks_label, 3), zticks_label, zticks     ))    
        zticks_label  = np.round( zticks_label, 3)
    return  zticks,zticks_label 




def show_qzr_map(  qr, qz, inc_x0, data=None, Nzline=10,Nrline=10  ):
    
    ''' 
    Dec 16, 2015, Y.G.@CHX
    plot a qzr map of a gisaxs image (data) 
    
    Parameters:
        qr:  2-D array, qr of a gisaxs image (data)
        qz:  2-D array, qz of a gisaxs image (data)
        inc_x0:  the incident beam center x 
         
    Options:
        data: 2-D array, a gisaxs image, if None, =qr+qz
        Nzline: int, z-line number
        Nrline: int, r-line number
        
        
    Return:
        zticks: list, z-tick positions in unit of pixel
        zticks_label: list, z-tick positions in unit of real space
        rticks: list, r-tick positions in unit of pixel
        rticks_label: list, r-tick positions in unit of real space
   
    
    Examples:
        
        ticks = show_qzr_map(  qr, qz, inc_x0, data = None, Nzline=10, Nrline= 10   )
        ticks = show_qzr_map(  qr,qz, inc_x0, data = avg_imgmr, Nzline=10,  Nrline=10   )
    '''
        
    
    import matplotlib.pyplot as plt    
    import copy
    import matplotlib.cm as mcm
    
    
    
    cmap='viridis'
    _cmap = copy.copy((mcm.get_cmap(cmap)))
    _cmap.set_under('w', 0)    
    
    
    qr_start, qr_end, qr_num = qr.min(),qr.max(), Nrline
    qz_start, qz_end, qz_num = qz.min(),qz.max(), Nzline 
    qr_edge, qr_center = get_qedge(qr_start , qr_end, ( qr_end- qr_start)/(qr_num+100), qr_num )
    qz_edge, qz_center = get_qedge( qz_start,   qz_end,   (qz_end - qz_start)/(qz_num+100 ) ,  qz_num )

    label_array_qz = get_qmap_label( qz, qz_edge)
    label_array_qr = get_qmap_label( qr, qr_edge)
 
    labels_qz, indices_qz = roi.extract_label_indices( label_array_qz  )
    labels_qr, indices_qr = roi.extract_label_indices( label_array_qr  )
    num_qz = len(np.unique( labels_qz ))
    num_qr = len(np.unique( labels_qr ))     


    fig, ax = plt.subplots(   figsize=(8,14)   )
    
    
    
    if data is None:
        data=qr+qz        
        im = ax.imshow(data, cmap='viridis',origin='lower') 
    else:
        im = ax.imshow(data, cmap='viridis',origin='lower',  norm= LogNorm(vmin=0.001, vmax=1e1)) 

    imr=ax.imshow(label_array_qr, origin='lower' ,cmap='viridis', vmin=0.5,vmax= None  )#,interpolation='nearest',) 
    imz=ax.imshow(label_array_qz, origin='lower' ,cmap='viridis', vmin=0.5,vmax= None )#,interpolation='nearest',) 

    #caxr = fig.add_axes([0.88, 0.2, 0.03, .7])  #x,y, width, heigth     
    #cba = fig.colorbar(im, cax=caxr    )      
    #cba = fig.colorbar(im,  fraction=0.046, pad=0.04)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    #fig.colorbar(im, shrink =.82)
    #cba = fig.colorbar(im)
    
    ax.set_xlabel(r'$q_r$', fontsize=18)
    ax.set_ylabel(r'$q_z$',fontsize=18)
 
    zticks,zticks_label  = get_qz_tick_label(qz,label_array_qz)
    #rticks,rticks_label  = get_qr_tick_label(label_array_qr,inc_x0)

    rticks,rticks_label = zip(*sorted(  zip( *get_qr_tick_label( qr, label_array_qr, inc_x0) ))  )

    #stride = int(len(zticks)/10)
    
    stride = 1
    ax.set_yticks( zticks[::stride] )
    yticks =  zticks_label[::stride] 
    ax.set_yticklabels(yticks, fontsize=7)

    #stride = int(len(rticks)/10)
    stride = 1
    ax.set_xticks( rticks[::stride] )
    xticks =  rticks_label[::stride]
    ax.set_xticklabels(xticks, fontsize=7)

    #print (xticks, yticks)
    

    ax.set_title( 'Q-zr_Map', y=1.03,fontsize=18)
    plt.show()    
    return  zticks,zticks_label,rticks,rticks_label



 
def show_qzr_roi( data, rois, inc_x0, ticks, alpha=0.3):    
        
    ''' 
    Dec 16, 2015, Y.G.@CHX
    plot a qzr map of a gisaxs image with rois( a label array) 
    
    Parameters:
        data: 2-D array, a gisaxs image 
        rois:  2-D array, a label array         
        inc_x0:  the incident beam center x 
        ticks:  zticks, zticks_label, rticks, rticks_label = ticks
            zticks: list, z-tick positions in unit of pixel
            zticks_label: list, z-tick positions in unit of real space
            rticks: list, r-tick positions in unit of pixel
            rticks_label: list, r-tick positions in unit of real space
         
    Options:
        alpha:  transparency of the label array on top of data
        
    Return:
         a plot of a qzr map of a gisaxs image with rois( a label array)
   
    
    Examples:        
        show_qzr_roi( avg_imgr, box_maskr, inc_x0, ticks)
         
    '''
      
        
        
    #import matplotlib.pyplot as plt    
    #import copy
    #import matplotlib.cm as mcm
    
    #cmap='viridis'
    #_cmap = copy.copy((mcm.get_cmap(cmap)))
    #_cmap.set_under('w', 0)
    zticks, zticks_label, rticks, rticks_label = ticks
    avg_imgr, box_maskr = data, rois
    num_qzr = len(np.unique( box_maskr)) -1
    fig, ax = plt.subplots(figsize=(8,12))
    ax.set_title("ROI--Labeled Array on Data")
    im,im_label = show_label_array_on_image(ax, avg_imgr, box_maskr, imshow_cmap='viridis',
                            cmap='Paired', alpha=alpha,
                             vmin=0.01, vmax=30. ,  origin="lower")


    for i in range( 1, num_qzr+1 ):
        ind =  np.where( box_maskr == i)[1]
        indz =  np.where( box_maskr == i)[0]
        c = '%i'%i
        y_val = int( indz.mean() )

        #print (ind[0], ind[1])
        if ind[0] < inc_x0 and ind[-1]>inc_x0:             
            x_val1 = int( (ind[np.where(ind < inc_x0)[0]]).mean() )
            x_val2 = int( (ind[np.where(ind > inc_x0)[0]]).mean() )
            ax.text(x_val1, y_val, c, va='center', ha='center')
            ax.text(x_val2, y_val, c, va='center', ha='center')

        else:
            x_val = int( ind.mean() )
            #print (xval, y)
            ax.text(x_val, y_val, c, va='center', ha='center')

        #print (x_val1,x_val2)

    #stride = int(len(zticks)/3)
    stride = 1
    ax.set_yticks( zticks[::stride] )
    yticks =  zticks_label[::stride] 
    ax.set_yticklabels(yticks, fontsize=9)

    #stride = int(len(rticks)/3)
    stride = 1
    ax.set_xticks( rticks[::stride] )
    xticks =  rticks_label[::stride]
    ax.set_xticklabels(xticks, fontsize=9)
    
    #print (xticks, yticks)

    #caxr = fig.add_axes([0.95, 0.1, 0.03, .8])  #x,y, width, heigth     
    #cba = fig.colorbar(im_label, cax=caxr    )  

    #fig.colorbar(im_label, shrink =.85)
    #fig.colorbar(im, shrink =.82)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    ax.set_xlabel(r'$q_r$', fontsize=22)
    ax.set_ylabel(r'$q_z$',fontsize=22)

    plt.show()

    
    
#plot g2 results

def plot_gisaxs_g2( g2, taus, res_pargs=None, *argv,**kwargs):     
    '''plot g2 results, 
       g2: one-time correlation function
       taus: the time delays  
       res_pargs, a dict, can contains
           uid/path/qr_center/qz_center/
       kwargs: can contains
           vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
           ylim/xlim: the limit of y and x
       
       e.g.
       plot_gisaxs_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )
           
      ''' 

    
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path'] 
        qz_center = res_pargs[ 'qz_center']
        num_qz = len( qz_center)
        qr_center = res_pargs[ 'qr_center']
        num_qr = len( qr_center)
    
    else:
    
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'
        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
        if 'qz_center' in kwargs.keys():
            qz_center = kwargs[ 'qz_center']
            num_qz = len( qz_center)
        else:
            print( 'Please give qz_center')
        if 'qr_center' in kwargs.keys():
            qr_center = kwargs[ 'qr_center']
            num_qr = len( qr_center)
        else:
            print( 'Please give qr_center') 


        

    for qz_ind in range(num_qz):
        fig = plt.figure(figsize=(10, 12))
        #fig = plt.figure()
        title_qz = ' Qz= %.5f  '%( qz_center[qz_ind]) + r'$\AA^{-1}$' 
        plt.title('uid= %s:--->'%uid + title_qz,fontsize=20, y =1.1) 
        #print (qz_ind,title_qz)
        if num_qz!=1:plt.axis('off')
        sx = int(round(np.sqrt(num_qr)) )
        if num_qr%sx == 0: 
            sy = int(num_qr/sx)
        else: 
            sy=int(num_qr/sx+1) 
        for sn in range(num_qr):
            ax = fig.add_subplot(sx,sy,sn+1 )
            ax.set_ylabel("g2") 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16) 

            title_qr = " Qr= " + '%.5f  '%( qr_center[sn]) + r'$\AA^{-1}$'
            if num_qz==1:

                title = 'uid= %s:--->'%uid + title_qz + '__' +  title_qr
            else:
                title = title_qr
            ax.set_title( title  )


            y=g2[:, sn + qz_ind * num_qr]
            ax.semilogx(taus, y, '-o', markersize=6)             
            
            if 'ylim' in kwargs:
                ax.set_ylim( kwargs['ylim'])
            elif 'vlim' in kwargs:
                vmin, vmax =kwargs['vlim']
                ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
            else:
                pass
            if 'xlim' in kwargs:
                ax.set_xlim( kwargs['xlim'])

        dt =datetime.now()
        CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)        
                
        fp = path + 'g2--uid=%s-qz=%s'%(uid,qz_center[qz_ind]) + CurTime + '.png'
        fig.savefig( fp, dpi=fig.dpi)        
        fig.tight_layout()  
        plt.show()

        
        
        
    
    
#plot g2 results

def plot_gisaxs_two_g2( g2, taus, g2b, tausb,res_pargs=None, *argv,**kwargs):        
    '''plot g2 results, 
       g2: one-time correlation function from a multi-tau method
       g2b: another g2 from a two-time method
       taus: the time delays        
       kwargs: can contains
           vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
           ylim/xlim: the limit of y and x
       
       e.g.
       plot_saxs_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, vlim=[.99, 1.01] )
           
      ''' 

    
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path'] 
        qz_center = res_pargs[ 'qz_center']
        num_qz = len( qz_center)
        qr_center = res_pargs[ 'qr_center']
        num_qr = len( qr_center)
    
    else:
    
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'
        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
        if 'qz_center' in kwargs.keys():
            qz_center = kwargs[ 'qz_center']
            num_qz = len( qz_center)
        else:
            print( 'Please give qz_center')
        if 'qr_center' in kwargs.keys():
            qr_center = kwargs[ 'qr_center']
            num_qr = len( qr_center)
        else:
            print( 'Please give qr_center') 


        

    for qz_ind in range(num_qz):
        fig = plt.figure(figsize=(10, 12))
        #fig = plt.figure()
        title_qz = ' Qz= %.5f  '%( qz_center[qz_ind]) + r'$\AA^{-1}$' 
        plt.title('uid= %s:--->'%uid + title_qz,fontsize=20, y =1.1) 
        #print (qz_ind,title_qz)
        if num_qz!=1:plt.axis('off')
        sx = int(round(np.sqrt(num_qr)) )
        if num_qr%sx == 0: 
            sy = int(num_qr/sx)
        else: 
            sy=int(num_qr/sx+1) 
        for sn in range(num_qr):
            ax = fig.add_subplot(sx,sy,sn+1 )
            ax.set_ylabel("g2") 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16) 

            title_qr = " Qr= " + '%.5f  '%( qr_center[sn]) + r'$\AA^{-1}$'
            if num_qz==1:

                title = 'uid= %s:--->'%uid + title_qz + '__' +  title_qr
            else:
                title = title_qr
            ax.set_title( title  )
            
            y=g2b[:, sn + qz_ind * num_qr]
            ax.semilogx( tausb, y, '--r', markersize=6,label= 'by-two-time')  

            y2=g2[:, sn]
            ax.semilogx(taus, y2, 'o', markersize=6,  label= 'by-multi-tau') 
        
            
            if sn + qz_ind * num_qr==0:
                ax.legend()
        
        
            if 'ylim' in kwargs:
                ax.set_ylim( kwargs['ylim'])
            elif 'vlim' in kwargs:
                vmin, vmax =kwargs['vlim']
                ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
            else:
                pass
            if 'xlim' in kwargs:
                ax.set_xlim( kwargs['xlim'])

        #dt =datetime.now()
        #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)        
                
        #fp = path + 'g2--uid=%s-qz=%s'%(uid,qz_center[qz_ind]) + CurTime + '.png'
        #fig.savefig( fp, dpi=fig.dpi)        
        fig.tight_layout()  
        plt.show()

        
        
        
        
        
def save_gisaxs_g2(  g2,res_pargs , *argv,**kwargs):     
    
    '''save g2 results, 
       res_pargs should contain
           g2: one-time correlation function
           res_pargs: contions taus, q_ring_center values
           path:
           uid:
      
      '''
    taus = res_pargs[ 'taus']
    qz_center = res_pargs['qz_center']
    qr_center = res_pargs['qr_center']
    path = res_pargs['path']
    uid = res_pargs['uid']
    
    df = DataFrame(     np.hstack( [ (taus).reshape( len(g2),1) ,  g2] )  ) 
    columns=[]
    columns.append('tau')
    for qz in qz_center:
        for qr in qr_center:
            columns.append( [str(qz),str(qr)] )
    df.columns = columns   
    
    dt =datetime.now()
    CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    
    filename = os.path.join(path, 'g2-%s-%s.csv' % (uid,CurTime))
    df.to_csv(filename)
    print( 'The g2 of uid= %s is saved in %s with filename as g2-%s-%s.csv'%(uid, path, uid, CurTime))

    
    

     
def stretched_auto_corr_scat_factor(lags, beta, relaxation_rate, alpha=1.0, baseline=1):
    return beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline

def simple_exponential(lags, beta, relaxation_rate,  baseline=1):
    return beta * np.exp(-2 * relaxation_rate * lags) + baseline


def fit_gisaxs_g2( g2, res_pargs, function='simple_exponential', *argv,**kwargs):     
    '''
    Fit one-time correlation function
    
    The support functions include simple exponential and stretched/compressed exponential
    Parameters
    ---------- 
    g2: one-time correlation function for fit, with shape as [taus, qs]
    res_pargs: a dict, contains keys
        taus: the time delay, with the same length as g2
        q_ring_center:  the center of q rings, for the title of each sub-plot
        uid: unique id, for the title of plot
        
    function: 
        'simple_exponential': fit by a simple exponential function, defined as  
                    beta * np.exp(-2 * relaxation_rate * lags) + baseline
        'streched_exponential': fit by a streched exponential function, defined as  
                    beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline
                    
    Returns
    -------        
    fit resutls:
        a dict, with keys as 
        'baseline': 
         'beta':
         'relaxation_rate': 
    an example:
        result = fit_g2( g2, res_pargs, function = 'simple')
        result = fit_g2( g2, res_pargs, function = 'stretched')
    '''
    
        
    taus = res_pargs[ 'taus']
    qz_center = res_pargs[ 'qz_center']
    num_qz = len( qz_center)
    qr_center = res_pargs[ 'qr_center']
    num_qr = len( qr_center)  
    uid=res_pargs['uid']    
    #uid=res_pargs['uid']  
    
    num_rings = g2.shape[1]
    beta = np.zeros(   num_rings )  #  contrast factor
    rate = np.zeros(   num_rings )  #  relaxation rate
    alpha = np.zeros(   num_rings )  #  alpha
    baseline = np.zeros(   num_rings )  #  baseline

        
    if function=='simple_exponential' or function=='simple':
        mod = Model(stretched_auto_corr_scat_factor,  independent_vars=['alpha', 'lags'])   
        
    elif function=='stretched_exponential' or function=='stretched':
        mod = Model(stretched_auto_corr_scat_factor,  independent_vars=[ 'lags'])   
    else:
        print ("The %s is not supported.The supported functions include simple_exponential and stretched_exponential"%function)
    
    
    
    
    for qz_ind in range(num_qz):
        fig = plt.figure(figsize=(10, 12))
        #fig = plt.figure()
        title_qz = ' Qz= %.5f  '%( qz_center[qz_ind]) + r'$\AA^{-1}$' 
        plt.title('uid= %s:--->'%uid + title_qz,fontsize=20, y =1.1) 
        #print (qz_ind,title_qz)
        if num_qz!=1:plt.axis('off')
        sx = int(round(np.sqrt(num_qr)) )
        if num_qr%sx == 0: 
            sy = int(num_qr/sx)
        else: 
            sy=int(num_qr/sx+1) 
        for sn in range(num_qr):
            
            ax = fig.add_subplot(sx,sy,sn+1 )
            ax.set_ylabel("g2") 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16) 

            title_qr = " Qr= " + '%.5f  '%( qr_center[sn]) + r'$\AA^{-1}$'
            if num_qz==1:

                title = 'uid= %s:--->'%uid + title_qz + '__' +  title_qr
            else:
                title = title_qr
            ax.set_title( title  )

            i = sn + qz_ind * num_qr            
            y=g2[1:, i]

            result1 = mod.fit(y, lags=taus[1:], beta=.05, alpha=1.0,
                          relaxation_rate =0.005, baseline=1.0,   )

            #print ( result1.best_values)
            rate[i] = result1.best_values['relaxation_rate']
            #rate[i] = 1e-16
            beta[i] = result1.best_values['beta'] 
        
            #baseline[i] = 1.0
            baseline[i] =  result1.best_values['baseline'] 

            if function=='simple_exponential' or function=='simple':
                alpha[i] =1.0 
            elif function=='stretched_exponential' or function=='stretched':
                alpha[i] = result1.best_values['alpha']

            ax.semilogx(taus[1:], y, 'ro')
            ax.semilogx(taus[1:], result1.best_fit, '-b')
             
            
            if 'ylim' in kwargs:
                ax.set_ylim( kwargs['ylim'])
            elif 'vlim' in kwargs:
                vmin, vmax =kwargs['vlim']
                ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
            else:
                pass
            if 'xlim' in kwargs:
                ax.set_xlim( kwargs['xlim'])                

            txts = r'$\gamma$' + r'$ = %.3f$'%(1/rate[i]) +  r'$ s$'
            ax.text(x =0.02, y=.55, s=txts, fontsize=14, transform=ax.transAxes)     
            txts = r'$\alpha$' + r'$ = %.3f$'%(alpha[i])  
            #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
            ax.text(x =0.02, y=.45, s=txts, fontsize=14, transform=ax.transAxes)  

            txts = r'$baseline$' + r'$ = %.3f$'%( baseline[i]) 
            ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)        
 
    
        #dt =datetime.now()
        #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute) 
        #fp = path + 'g2--uid=%s-qz=%s-fit'%(uid,qz_center[qz_ind]) + CurTime + '.png'
        #fig.savefig( fp, dpi=fig.dpi)        
        
    fig.tight_layout()  
    plt.show()
    result = dict( beta=beta, rate=rate, alpha=alpha, baseline=baseline )
    
    return result

   
    
    
    
    
#GiSAXS End
###############################


    
def get_each_box_mean_intensity( data_series, box_mask, sampling, timeperframe, plot_ = True ,  *argv,**kwargs):   
    
    
    mean_int_sets, index_list = roi.mean_intensity(np.array(data_series[::sampling]), box_mask) 
    try:
        N = len(data_series)
    except:
        N = data_series.length
    times = np.arange( N )*timeperframe  # get the time for each frame
    num_rings = len( np.unique( box_mask)[1:] )
    if plot_:
        fig, ax = plt.subplots(figsize=(8, 8))
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
            
        ax.set_title("Uid= %s--Mean intensity of each box"%uid)
        for i in range(num_rings):
            ax.plot( times[::sampling], mean_int_sets[:,i], label="Box "+str(i+1),marker = 'o', ls='-')
            ax.set_xlabel("Time")
            ax.set_ylabel("Mean Intensity")
        ax.legend() 
        plt.show()
    return times, mean_int_sets










