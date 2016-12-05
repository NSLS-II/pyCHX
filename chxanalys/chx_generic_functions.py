from chxanalys.chx_libs import *
#from tqdm import *



def create_cross_mask(  image, center, wy_left=4, wy_right=4, wx_up=4, wx_down=4,
                     center_circle = True, center_radius=10
                     ):
    '''
    Give image and the beam center to create a cross-shaped mask
    wy_left: the width of left h-line
    wy_right: the width of rigth h-line
    wx_up: the width of up v-line
    wx_down: the width of down v-line
    center_circle: if True, create a circle with center and center_radius
    
    Return:
    the cross mask
    '''
    from skimage.draw import line_aa, line, polygon, circle
    
    imy, imx = image.shape   
    cx,cy = center
    bst_mask = np.zeros_like( image , dtype = bool)   
    ###
    #for right part    
    wy = wy_right
    x = np.array( [ cx, imx, imx, cx  ])  
    y = np.array( [ cy-wy, cy-wy, cy + wy, cy + wy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1
    
    ###
    #for left part    
    wy = wy_left
    x = np.array( [0,  cx, cx,0  ])  
    y = np.array( [ cy-wy, cy-wy, cy + wy, cy + wy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1    
    
    ###
    #for up part    
    wx = wx_up
    x = np.array( [ cx-wx, cx + wx, cx+wx, cx-wx  ])  
    y = np.array( [ cy, cy, imy, imy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1    
    
    ###
    #for low part    
    wx = wx_down
    x = np.array( [ cx-wx, cx + wx, cx+wx, cx-wx  ])  
    y = np.array( [ 0,0, cy, cy])
    rr, cc = polygon( y,x)
    bst_mask[rr,cc] =1   
    
    rr, cc = circle( cy, cx, center_radius)
    bst_mask[rr,cc] =1   
    
    
    full_mask= ~bst_mask
    
    return full_mask
    
    
    


def generate_edge( centers, width):
    '''YG. 10/14/2016
    give centers and width (number or list) to get edges'''
    edges = np.zeros( [ len(centers),2])
    edges[:,0] =  centers - width
    edges[:,1] =  centers + width
    return edges


def export_scan_scalar( uid, x='dcm_b', y= ['xray_eye1_stats1_total'],
                       path='/XF11ID/analysis/2016_3/commissioning/Results/' ):
    '''YG. 10/17/2016
    export uid data to a txt file
    uid: unique scan id
    x: the x-col 
    y: the y-cols
    path: save path
    Example:
        data = export_scan_scalar( uid, x='dcm_b', y= ['xray_eye1_stats1_total'],
                       path='/XF11ID/analysis/2016_3/commissioning/Results/exported/' )
    A plot for the data:
        d.plot(x='dcm_b', y = 'xray_eye1_stats1_total', marker='o', ls='-', color='r')
        
    '''
    from databroker import DataBroker as db, get_images, get_table, get_events, get_fields 
    from chxanalys.chx_generic_functions import  trans_data_to_pd
    
    hdr = db[uid]
    print( get_fields( hdr ) )
    data = get_table( db[uid] )
    xp = data[x]
    datap = np.zeros(  [len(xp), len(y)+1])
    datap[:,0] = xp
    for i, yi in enumerate(y):
        datap[:,i+1] = data[yi]
        
    datap = trans_data_to_pd( datap, label=[x] + [yi for yi in y])   
    datap.to_csv( path + 'uid=%s.csv'%uid)
    return datap




#####
#load data by databroker   

def get_flatfield( uid, reverse=False ):
    import h5py
    detector = get_detector( db[uid ] )
    sud = get_sid_filenames(db[uid])
    master_path = '%s_master.h5'%(sud[2][0])
    print( master_path)
    f= h5py.File(master_path, 'r')
    k= 'entry/instrument/detector/detectorSpecific/' #data_collection_date'
    d= np.array( f[ k]['flatfield'] )
    f.close()
    if reverse:        
        d = reverse_updown( d )
        
    return d



def get_detector( header ):
    keys = [k for k, v in header.descriptors[0]['data_keys'].items()     if 'external' in v]
    return keys[0]

    
def get_sid_filenames(header, fill=True):
    """get a bluesky scan_id, unique_id, filename by giveing uid and detector
        
    Parameters
    ----------
    header: a header of a bluesky scan, e.g. db[-1]
        
    Returns
    -------
    scan_id: integer
    unique_id: string, a full string of a uid
    filename: sring
    
    Usuage:
    sid,uid, filenames   = get_sid_filenames(db[uid])
    
    """   
    
    keys = [k for k, v in header.descriptors[0]['data_keys'].items()     if 'external' in v]
    events = get_events( header, keys, handler_overrides={key: RawHandler for key in keys}, fill=fill)
    key, = keys   
    filenames =  [  str( ev['data'][key][0]) + '_'+ str(ev['data'][key][2]['seq_id']) for ev in events]     
    sid = header['start']['scan_id']
    uid= header['start']['uid']
    
    return sid,uid, filenames   


def load_data( uid , detector = 'eiger4m_single_image', fill=True, reverse=False):
    """load bluesky scan data by giveing uid and detector
        
    Parameters
    ----------
    uid: unique ID of a bluesky scan
    detector: the used area detector
    fill: True to fill data
    reverse: if True, reverse the image upside down to match the "real" image geometry (should always be True in the future)
    
    Returns
    -------
    image data: a pims frames series
    if not success read the uid, will return image data as 0
    
    Usuage:
    imgs = load_data( uid, detector  )
    md = imgs.md
    """   
    hdr = db[uid]
    flag =1
    while flag<2 and flag !=0:    
        try:
            ev, = get_events(hdr, [detector], fill=fill) 
            flag = 0 
            
        except:
            flag += 1        
            print ('Trying again ...!')

         
    if flag:
        try:
            imgs = get_images( hdr, detector)
            if len(imgs[0])==1:
                md = imgs[0].md
                imgs = pims.pipeline(lambda img:  img[0])(imgs)
                imgs.md = md
        except:            
            print ("Can't Load Data!")
            uid = '00000'  #in case of failling load data
            imgs = 0
    else:
        imgs = ev['data'][detector]
    #print (imgs)
    if reverse:
        md=imgs.md
        imgs = reverse_updown( imgs )
        imgs.md = md
    return imgs



def load_data2( uid , detector = 'eiger4m_single_image'  ):
    """load bluesky scan data by giveing uid and detector
        
    Parameters
    ----------
    uid: unique ID of a bluesky scan
    detector: the used area detector
    
    Returns
    -------
    image data: a pims frames series
    if not success read the uid, will return image data as 0
    
    Usuage:
    imgs = load_data( uid, detector  )
    md = imgs.md
    """   
    hdr = db[uid]
    flag =1
    while flag<4 and flag !=0:    
        try:
            ev, = get_events(hdr, [detector]) 
            flag =0 
        except:
            flag += 1        
            print ('Trying again ...!')

    if flag:
        print ("Can't Load Data!")
        uid = '00000'  #in case of failling load data
        imgs = 0
    else:
        imgs = ev['data'][detector]

    #print (imgs)
    return imgs



def psave_obj(obj, filename ):
    '''save an object with filename by pickle.dump method
       This function automatically add '.pkl' as filename extension
    Input:
        obj: the object to be saved
        filename: filename (with full path) to be saved
    Return:
        None    
    '''
    with open( filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pload_obj(filename ):
    '''load a pickled filename 
        This function automatically add '.pkl' to filename extension
    Input:        
        filename: filename (with full path) to be saved
    Return:
        load the object by pickle.load method
    '''
    with open( filename + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
    
def load_mask( path, mask_name, plot_ = False, reverse=False, *argv,**kwargs): 
    
    """load a mask file
    the mask is a numpy binary file (.npy) 
        
    Parameters
    ----------
    path: the path of the mask file
    mask_name: the name of the mask file
    plot_: a boolen type
    reverse: if True, reverse the image upside down to match the "real" image geometry (should always be True in the future)    
    Returns
    -------
    mask: array
    if plot_ =True, will show the mask   
    
    Usuage:
    mask = load_mask( path, mask_name, plot_ =  True )
    """
    
    mask = np.load(    path +   mask_name )
    mask = np.array(mask, dtype = np.int32)
    if reverse:
        mask = mask[::-1,:]  
    if plot_:
        show_img( mask, *argv,**kwargs)   
    return mask



def create_hot_pixel_mask(img, threshold, center=None, center_radius=300 ):
    '''create a hot pixel mask by giving threshold
       Input:
           img: the image to create hot pixel mask
           threshold: the threshold above which will be considered as hot pixels
           center: optional, default=None
                             else, as a two-element list (beam center), i.e., [center_x, center_y]
           if center is not None, the hot pixel will not include a circle region 
                           which is defined by center and center_radius ( in unit of pixel)
       Output:
           a bool types numpy array (mask), 1 is good and 0 is excluded   
    
    '''
    bst_mask = np.ones_like( img , dtype = bool)    
    if center is not None:    
        from skimage.draw import  circle    
        imy, imx = img.shape   
        cy,cx = center        
        rr, cc = circle( cy, cx, center_radius)
        bst_mask[rr,cc] =0 
    
    hmask = np.ones_like( img )
    hmask[np.where( img * bst_mask  > threshold)]=0
    return hmask




def apply_mask( imgs, mask):
    '''apply mask to imgs to produce a generator
    
    Usuages:
    imgsa = apply_mask( imgs, mask )
    good_series = apply_mask( imgs[good_start:], mask )
    
    '''
    return pims.pipeline(lambda img: np.int_(mask) * img)(imgs)  # lazily apply mask


def reverse_updown( imgs):
    '''reverse imgs upside down to produce a generator
    
    Usuages:
    imgsr = reverse_updown( imgs)
     
    
    '''
    return pims.pipeline(lambda img: img[::-1,:])(imgs)  # lazily apply mask


def RemoveHot( img,threshold= 1E7, plot_=True ):
    '''Remove hot pixel from img'''
    
    mask = np.ones_like( np.array( img )    )
    badp = np.where(  np.array(img) >= threshold )
    if len(badp[0])!=0:                
        mask[badp] = 0  
    if plot_:
        show_img( mask )
    return mask


############
###plot data


def show_img( image, ax=None,xlim=None, ylim=None, save=False,image_name=None,path=None, 
             aspect=None, logs=False,vmin=None,vmax=None,return_fig=False,cmap='viridis', 
             show_time= False, file_name =None,
             *argv,**kwargs ):    
    """a simple function to show image by using matplotlib.plt imshow
    pass *argv,**kwargs to imshow
    
    Parameters
    ----------
    image : array
        Image to show
    Returns
    -------
    None
    """ 
    
    if ax is None:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
    else:
        fig, ax=ax
        
    if not logs:
        #im=ax.imshow(img, origin='lower' ,cmap='viridis',interpolation="nearest" , vmin=vmin,vmax=vmax)
        im=ax.imshow(image, origin='lower' ,cmap=cmap,interpolation="nearest", vmin=vmin,vmax=vmax)  #vmin=0,vmax=1,
    else:
        im=ax.imshow(image, origin='lower' ,cmap=cmap,
        interpolation="nearest" , norm=LogNorm(vmin,  vmax))          
        
    fig.colorbar(im)
    ax.set_title( image_name )
    if xlim is not None:
        ax.set_xlim(   xlim  )
    if ylim is not None:
        ax.set_ylim(   ylim )
    if aspect is not None:
        #aspect = image.shape[1]/float( image.shape[0] )
        ax.set_aspect(aspect)
    else:
        ax.set_aspect(aspect='auto')
    if save:
        if show_time:
            dt =datetime.now()
            CurTime = '_%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)         
            fp = path + '%s'%( file_name ) + CurTime + '.png'       
        else:
            fp = path + '%s'%( image_name ) + '.png'
            
        plt.savefig( fp, dpi=fig.dpi)        
    #plt.show()
    if return_fig:
        return fig
    
    
def plot1D( y,x=None, ax=None,return_fig=False,*argv,**kwargs):    
    """a simple function to plot two-column data by using matplotlib.plot
    pass *argv,**kwargs to plot
    
    Parameters
    ----------
    y: column-y
    x: column-x, by default x=None, the plot will use index of y as x-axis
    Returns
    -------
    None
    """     
    if ax is None:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
        
    if 'legend' in kwargs.keys():
        legend =  kwargs['legend']  
    else:
        legend = ' '

    try:
         logx = kwargs['logx']
    except:
        logx=False
    try:
         logy = kwargs['logy']
    except:
        logy=False
        
    try:
         logxy = kwargs['logxy']
    except:
        logxy= False        

    if logx==True and logy==True:
        logxy = True
        
    if x is None:
        ax.plot( y, marker = 'o', ls='-',label= legend)#,*argv,**kwargs)
        if logx:
            ax.semilogx( y, marker = 'o', ls='-')#,*argv,**kwargs)
        if logy:
            ax.semilogy( y, marker = 'o', ls='-')#,*argv,**kwargs)
        if logxy:
            ax.loglog( y, marker = 'o', ls='-')#,*argv,**kwargs)
    else:
        ax.plot(x,y, marker='o',ls='-',label= legend)#,*argv,**kwargs)        
        if logx:
            ax.semilogx( x,y, marker = 'o', ls='-')#,*argv,**kwargs)
        if logy:
            ax.semilogy( x,y, marker = 'o', ls='-')#,*argv,**kwargs)
        if logxy:
            ax.loglog( x,y, marker = 'o', ls='-')#,*argv,**kwargs) 
    
    if 'xlim' in kwargs.keys():
         ax.set_xlim(    kwargs['xlim']  )    
    if 'ylim' in kwargs.keys():
         ax.set_ylim(    kwargs['ylim']  )
    if 'xlabel' in kwargs.keys():            
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys():            
        ax.set_ylabel(kwargs['ylabel'])
        
    if 'title' in kwargs.keys():
        title =  kwargs['title']
    else:
        title =  'plot'
    ax.set_title( title ) 
    #ax.set_xlabel("$Log(q)$"r'($\AA^{-1}$)')    
    ax.legend(loc = 'best')  
    if 'save' in kwargs.keys():
        if  kwargs['save']: 
            #dt =datetime.now()
            #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)         
            #fp = kwargs['path'] + '%s'%( title ) + CurTime + '.png'  
            fp = kwargs['path'] + '%s'%( title )   + '.png' 
            plt.savefig( fp, dpi=fig.dpi)         
    if return_fig:
        return fig      

        
###

def check_shutter_open( data_series,  min_inten=0, time_edge = [0,10], plot_ = False,  *argv,**kwargs):     
      
    '''Check the first frame with shutter open
    
        Parameters        
        ----------
        data_series: a image series
        min_inten: the total intensity lower than min_inten is defined as shtter close
        time_edge: the searching frame number range
        
        return:
        shutter_open_frame: a integer, the first frame number with open shutter     
        
        Usuage:
        good_start = check_shutter_open( imgsa,  min_inten=5, time_edge = [0,20], plot_ = False )
       
    '''
    imgsum =  np.array(  [np.sum(img ) for img in data_series[time_edge[0]:time_edge[1]:1]]  ) 
    if plot_:
        fig, ax = plt.subplots()  
        ax.plot(imgsum,'bo')
        ax.set_title('uid=%s--imgsum'%uid)
        ax.set_xlabel( 'Frame' )
        ax.set_ylabel( 'Total_Intensity' ) 
        plt.show()       
    shutter_open_frame = np.where( np.array(imgsum) > min_inten )[0][0]
    print ('The first frame with open shutter is : %s'%shutter_open_frame )
    return shutter_open_frame



def get_each_frame_intensity( data_series, sampling = 50, 
                             bad_pixel_threshold=1e10,                              
                             plot_ = False, save= False, *argv,**kwargs):   
    '''Get the total intensity of each frame by sampling every N frames
       Also get bad_frame_list by check whether above  bad_pixel_threshold  
       
       Usuage:
       imgsum, bad_frame_list = get_each_frame_intensity(good_series ,sampling = 1000, 
                                bad_pixel_threshold=1e10,  plot_ = True)
    '''
    
    #print ( argv, kwargs )
    imgsum =  np.array(  [np.sum(img ) for img in tqdm( data_series[::sampling] , leave = True ) ] )  
    if plot_:
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid']        
        fig, ax = plt.subplots()  
        ax.plot(imgsum,'bo')
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
            #fp = path + "Uid= %s--Waterfall-"%uid + CurTime + '.png'     
            fp = path + "uid=%s--imgsum-"%uid  + '.png'    
            fig.savefig( fp, dpi=fig.dpi)        
        plt.show()  
        
    bad_frame_list = np.where( np.array(imgsum) > bad_pixel_threshold )[0]
    if len(bad_frame_list):
        print ('Bad frame list are: %s' %bad_frame_list)
    else:
        print ('No bad frames are involved.')
    return imgsum,bad_frame_list


 

def create_time_slice( N, slice_num, slice_width, edges=None ):
    '''create a ROI time regions '''
    if edges is not None:
        time_edge = edges
    else:
        if slice_num==1:
            time_edge =  [ [0,N] ]
        else:
            tstep = N // slice_num
            te = np.arange( 0, slice_num +1   ) * tstep
            tc = np.int_( (te[:-1] + te[1:])/2 )[1:-1]
            if slice_width%2:
                sw = slice_width//2 +1
                time_edge =   [ [0,slice_width],  ] + [  [s-sw+1,s+sw] for s in tc  ] +  [ [N-slice_width,N]]
            else:
                sw= slice_width//2
                time_edge =   [ [0,slice_width],  ] + [  [s-sw,s+sw] for s in tc  ] +  [ [N-slice_width,N]]
                                 
            

    return time_edge


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
    
    
    
def show_ROI_on_image( image, ROI, center=None, rwidth=400,alpha=0.3,  label_on = True,
                       save=False, return_fig = False, rect_reqion=None,
                       *argv,**kwargs):
    
    '''show ROI on an image
        image: the data frame
        ROI: the interested region
        center: the plot center
        rwidth: the plot range around the center  
    
    '''
    
    vmin=0.01    
    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
        
    vmax=5
    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
        
    if RUN_GUI:
        fig = Figure(figsize=(8,8))
        axes = fig.add_subplot(111)
    else:
        fig, axes = plt.subplots(figsize=(8,8))
        
    axes.set_title("ROI on Image")
    im,im_label = show_label_array_on_image(axes, image, ROI, imshow_cmap='viridis',
                            cmap='Paired',alpha=alpha,
                             vmin=vmin, vmax=vmax,  origin="lower")
    if rect_reqion is  None:
        if center is not None:
            x1,x2 = [center[1] - rwidth, center[1] + rwidth]
            y1,y2 = [center[0] - rwidth, center[0] + rwidth]
            axes.set_xlim( [x1,x2])
            axes.set_ylim( [y1,y2])
    else:
        x1,x2,y1,y2= rect_reqion
        axes.set_xlim( [x1,x2])
        axes.set_ylim( [y1,y2])
    
    if label_on:
        num_qzr = len(np.unique( ROI )) -1        
        for i in range( 1, num_qzr + 1 ):
            ind =  np.where( ROI == i)[1]
            indz =  np.where( ROI == i)[0]
            c = '%i'%i
            y_val = int( indz.mean() )
            x_val = int( ind.mean() )
            #print (xval, y)
            axes.text(x_val, y_val, c, va='center', ha='center')         
    #fig.colorbar(im_label)
    fig.colorbar(im)
    if save:
        #dt =datetime.now()
        #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
        path = kwargs['path'] 
        if 'uid' in kwargs:
            uid = kwargs['uid']
        else:
            uid = 'uid'
        #fp = path + "Uid= %s--Waterfall-"%uid + CurTime + '.png'     
        fp = path + "uid=%s--ROI-on-Image-"%uid  + '.png'    
        plt.savefig( fp, dpi=fig.dpi)      
    #plt.show()
    if return_fig:
        return fig, axes, im 

        

        
def crop_image(  image,  crop_mask  ):
    
    ''' Crop the non_zeros pixels of an image  to a new image 
         
         
    '''
    from skimage.util import crop, pad     
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
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
    crops = crop_mask*image
    img_crop = crop(  crops, ((minpixelx, imgwidthx -  maxpixelx -1 ),
                                (minpixely, imgwidthy -  maxpixely -1 )) )
    return img_crop
            

def get_avg_img( data_series, sampling = 100, plot_ = False , save=False, *argv,**kwargs):   
    '''Get average imagef from a data_series by every sampling number to save time'''
    avg_img = np.average(data_series[:: sampling], axis=0)
    if plot_:
        fig, ax = plt.subplots()
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
            
        im = ax.imshow(avg_img , cmap='viridis',origin='lower',
                   norm= LogNorm(vmin=0.001, vmax=1e2))
        #ax.set_title("Masked Averaged Image")
        ax.set_title('uid= %s--Masked Averaged Image'%uid)
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
            fig.savefig( fp, dpi=fig.dpi)        
        plt.show()

    return avg_img



def check_ROI_intensity( avg_img, ring_mask, ring_number=3 , save=False, *argv,**kwargs):
    
    """plot intensity versus pixel of a ring        
    Parameters
    ----------
    avg_img: 2D-array, the image
    ring_mask: 2D-array  
    ring_number: which ring to plot
    
    Returns
    -------

     
    """   
    uid = 'uid'
    if 'uid' in kwargs.keys():
        uid = kwargs['uid'] 
    pixel = roi.roi_pixel_values(avg_img, ring_mask, [ring_number] )
    fig, ax = plt.subplots()
    ax.set_title('uid= %s--check-RIO-%s-intensity'%(uid, ring_number) )
    ax.plot( pixel[0][0] ,'bo', ls='-' )
    ax.set_ylabel('Intensity')
    ax.set_xlabel('pixel')
    if save:
        #dt =datetime.now()
        #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
        path = kwargs['path'] 
        #fp = path + "Uid= %s--Waterfall-"%uid + CurTime + '.png'     
        #fp = path + "uid=%s--check-ROI-intensity-"%uid  + '.png'  
        fp = path + "uid= %s--Mean-intensity-of-each-ROI-"%uid  + '.png' 
        fig.savefig( fp, dpi=fig.dpi)  
    
    plt.show()
    return pixel[0][0]

#from tqdm import tqdm

def cal_g2( image_series, ring_mask, bad_image_process,
           bad_frame_list=None,good_start=0, num_buf = 8, num_lev = None ):
    '''calculation g2 by using a multi-tau algorithm'''
    
    noframes = len( image_series)  # number of frames, not "no frames"
    #num_buf = 8  # number of buffers

    if bad_image_process:   
        import skbeam.core.mask as mask_image       
        bad_img_list = np.array( bad_frame_list) - good_start
        new_imgs = mask_image.bad_to_nan_gen( image_series, bad_img_list)        

        if num_lev is None:
            num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
        print ('In this g2 calculation, the buf and lev number are: %s--%s--'%(num_buf,num_lev))
        print ('%s frames will be processed...'%(noframes))
        print( 'Bad Frames involved!')

        g2, lag_steps = corr.multi_tau_auto_corr(num_lev, num_buf,  ring_mask, tqdm( new_imgs) )
        print( 'G2 calculation DONE!')

    else:

        if num_lev is None:
            num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
        print ('In this g2 calculation, the buf and lev number are: %s--%s--'%(num_buf,num_lev))
        print ('%s frames will be processed...'%(noframes))
        g2, lag_steps = corr.multi_tau_auto_corr(num_lev, num_buf,   ring_mask, tqdm(image_series) )
        print( 'G2 calculation DONE!')
        
    return g2, lag_steps



def run_time(t0):
    '''Calculate running time of a program
    Parameters
    ----------
    t0: time_string, t0=time.time()
        The start time
    Returns
    -------
    Print the running time 
    
    One usage
    ---------
    t0=time.time()
    .....(the running code)
    run_time(t0)
    '''    
    
    elapsed_time = time.time() - t0
    print ('Total time: %.2f min' %(elapsed_time/60.)) 
    
    
def trans_data_to_pd(data, label=None,dtype='array'):
    '''
    convert data into pandas.DataFrame
    Input:
        data: list or np.array
        label: the coloum label of the data
        dtype: list or array
    Output:
        a pandas.DataFrame
    '''
    #lists a [ list1, list2...] all the list have the same length
    from numpy import arange,array
    import pandas as pd,sys    
    if dtype == 'list':
        data=array(data).T        
    elif dtype == 'array':
        data=array(data)        
    else:
        print("Wrong data type! Now only support 'list' and 'array' tpye")        
    N,M=data.shape    
    index =  arange( N )
    if label is None:label=['data%s'%i for i in range(M)]
    #print label
    df = pd.DataFrame( data, index=index, columns= label  )
    return df


def save_lists( data, label=None,  filename=None, path=None):    
    '''
    save_lists( data, label=None,  filename=None, path=None)
    
    save lists to a CSV file with filename in path
    Parameters
    ----------
    data: list
    label: the column name, the length should be equal to the column number of list
    filename: the filename to be saved
    path: the filepath to be saved
    
    Example:    
    save_arrays(  [q,iq], label= ['q_A-1', 'Iq'], filename='uid=%s-q-Iq'%uid, path= data_dir  )    
    '''
    
    M,N = len(data[0]),len(data)
    d = np.zeros( [N,M] )
    for i in range(N):
        d[i] = data[i]        
    
    df = trans_data_to_pd(d.T, label, 'array')  
    #dt =datetime.now()
    #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    if filename is None:
        filename = 'data'
    filename = os.path.join(path, filename +'.csv')
    df.to_csv(filename)

def get_pos_val_overlap( p1, v1, p2,v2, Nl):
    '''get the overlap of v1 and v2
        p1: the index of array1 in array with total length as Nl
        v1: the corresponding value of p1
        p2: the index of array2 in array with total length as Nl
        v2: the corresponding value of p2
        Return:
         The values in v1 with the position in overlap of p1 and p2
         The values in v2 with the position in overlap of p1 and p2
         
         An example:
            Nl =10
            p1= np.array( [1,3,4,6,8] )
            v1 = np.array( [10,20,30,40,50])
            p2= np.array( [ 0,2,3,5,7,8])
            v2=np.array( [10,20,30,40,50,60,70])
            
            get_pos_val_overlap( p1, v1, p2,v2, Nl)
         
    '''
    ind = np.zeros( Nl, dtype=np.int32 )
    ind[p1] = np.arange( len(p1) ) +1 
    w2 = np.where( ind[p2] )[0]
    w1 = ind[ p2[w2]] -1
    return v1[w1], v2[w2]
    
    
    
def save_arrays( data, label=None, dtype='array', filename=None, path=None):    
    '''
    July 10, 2016, Y.G.@CHX
    save_arrays( data, label=None, dtype='array', filename=None, path=None): 
    save data to a CSV file with filename in path
    Parameters
    ----------
    data: arrays
    label: the column name, the length should be equal to the column number of data
    dtype: array or list
    filename: the filename to be saved
    path: the filepath to be saved
    
    Example:
    
    save_arrays(  qiq, label= ['q_A-1', 'Iq'], dtype='array', filename='uid=%s-q-Iq'%uid, path= data_dir  )

    
    '''
    df = trans_data_to_pd(data, label,dtype)  
    #dt =datetime.now()
    #CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    if filename is None:
        filename = 'data'
    filename = os.path.join(path, filename +'.csv')
    df.to_csv(filename)
    #print( 'The g2 of uid= %s is saved in %s with filename as g2-%s-%s.csv'%(uid, path, uid, CurTime))

def get_diffusion_coefficient( visocity, radius, T=298):
    '''July 10, 2016, Y.G.@CHX
        get diffusion_coefficient of a Brownian motion particle with radius in fuild with visocity
        visocity: N*s/m^2
        radius: m
        T: K
        k: 1.38064852(79)×10−23 J/T, Boltzmann constant   
        
        return diffusion_coefficient in unit of A^2/s
        e.g., for a 250 nm sphere in glycerol/water (90:10) at RT (298K) gives:
       1.38064852*10**(−23) *298  / ( 6*np.pi* 0.20871 * 250 *10**(-9)) * 10**20 /1e5 = 4.18*10^5 A2/s
       
       get_diffusion_coefficient( 0.20871, 250 *10**(-9), T=298) 
       
    '''
    
    k=  1.38064852*10**(-23)    
    return k*T / ( 6*np.pi* visocity * radius) * 10**20 




def ring_edges(inner_radius, width, spacing=0, num_rings=None):
    """
    Aug 02, 2016, Y.G.@CHX
    ring_edges(inner_radius, width, spacing=0, num_rings=None)
    
    Calculate the inner and outer radius of a set of rings.

    The number of rings, their widths, and any spacing between rings can be
    specified. They can be uniform or varied.

    Parameters
    ----------
    inner_radius : float
        inner radius of the inner-most ring

    width : float or list of floats
        ring thickness
        If a float, all rings will have the same thickness.

    spacing : float or list of floats, optional
        margin between rings, 0 by default
        If a float, all rings will have the same spacing. If a list,
        the length of the list must be one less than the number of
        rings.

    num_rings : int, optional
        number of rings
        Required if width and spacing are not lists and number
        cannot thereby be inferred. If it is given and can also be
        inferred, input is checked for consistency.

    Returns
    -------
    edges : array
        inner and outer radius for each ring

    Example
    -------
    # Make two rings starting at r=1px, each 5px wide
    >>> ring_edges(inner_radius=1, width=5, num_rings=2)
    [(1, 6), (6, 11)]
    # Make three rings of different widths and spacings.
    # Since the width and spacings are given individually, the number of
    # rings here is simply inferred.
    >>> ring_edges(inner_radius=1, width=(5, 4, 3), spacing=(1, 2))
    [(1, 6), (7, 11), (13, 16)]
    
    """
    # All of this input validation merely checks that width, spacing, and
    # num_rings are self-consistent and complete.
    width_is_list = isinstance(width, collections.Iterable)
    spacing_is_list = isinstance(spacing, collections.Iterable)
    if (width_is_list and spacing_is_list):
        if len(width) != len(spacing) + 1:
            raise ValueError("List of spacings must be one less than list "
                             "of widths.")
    if num_rings is None:
        try:
            num_rings = len(width)
        except TypeError:
            try:
                num_rings = len(spacing) + 1
            except TypeError:
                raise ValueError("Since width and spacing are constant, "
                                 "num_rings cannot be inferred and must be "
                                 "specified.")
    else:
        if width_is_list:
            if num_rings != len(width):
                raise ValueError("num_rings does not match width list")
        if spacing_is_list:
            if num_rings-1 != len(spacing):
                raise ValueError("num_rings does not match spacing list")

    # Now regularlize the input.
    if not width_is_list:
        width = np.ones(num_rings) * width
    if not spacing_is_list:
        spacing = np.ones(num_rings - 1) * spacing

    # The inner radius is the first "spacing."
    all_spacings = np.insert(spacing, 0, inner_radius)
    steps = np.array([all_spacings, width]).T.ravel()
    edges = np.cumsum(steps).reshape(-1, 2)

    return edges



def get_non_uniform_edges(  centers, width = 4, number_rings=3 ):
    '''
    YG CHX Spe 6
    get_non_uniform_edges(  centers, width = 4, number_rings=3 )
    
    Calculate the inner and outer radius of a set of non uniform distributed
    rings by giving ring centers
    For each center, there are number_rings with each of width

    Parameters
    ----------
    centers : float
        the center of the rings

    width : float or list of floats
        ring thickness
        If a float, all rings will have the same thickness.

    num_rings : int, optional
        number of rings
        Required if width and spacing are not lists and number
        cannot thereby be inferred. If it is given and can also be
        inferred, input is checked for consistency.

    Returns
    -------
    edges : array
        inner and outer radius for each ring    
    '''
    
    
    
    edges = np.zeros( [len(centers)*number_rings, 2]  )
    for i, c in enumerate(centers):       
        edges[i*number_rings:(i+1)*number_rings,:] = ring_edges( inner_radius =  c - width*number_rings/2,  
                      width=width, spacing= 0, num_rings=number_rings)
    return edges   



def trans_tf_to_td(tf, dtype = 'dframe'):
    '''July 02, 2015, Y.G.@CHX
    Translate epoch time to string
    '''
    import pandas as pd
    import numpy as np
    import datetime
    '''translate time.float to time.date,
       td.type dframe: a dataframe
       td.type list,   a list
    ''' 
    if dtype is 'dframe':ind = tf.index
    else:ind = range(len(tf))    
    td = np.array([ datetime.datetime.fromtimestamp(tf[i]) for i in ind ])
    return td



def trans_td_to_tf(td, dtype = 'dframe'):
    '''July 02, 2015, Y.G.@CHX
    Translate string to epoch time
    
    '''
    import time
    import numpy as np
    '''translate time.date to time.float,
       td.type dframe: a dataframe
       td.type list,   a list
    ''' 
    if dtype is 'dframe':ind = td.index
    else:ind = range(len(td))
    #tf = np.array([ time.mktime(td[i].timetuple()) for i in range(len(td)) ])
    tf = np.array([ time.mktime(td[i].timetuple()) for i in ind])
    return tf






















        