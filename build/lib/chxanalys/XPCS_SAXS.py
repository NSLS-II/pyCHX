from .chx_generic_functions import *


    
   
    
    
def circular_average(image, calibrated_center, threshold=0, nx=1500,
                     pixel_size=(1, 1),  min_x=None, max_x=None, mask=None):
    """Circular average of the the image data
    The circular average is also known as the radial integration
    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    calibrated_center : tuple
        The center of the image in pixel units
        argument order should be (row, col)
    threshold : int, optional
        Ignore counts above `threshold`
        default is zero
    nx : int, optional
        number of bins in x
        defaults is 100 bins
    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)
    min_x : float, optional number of pixels
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional number of pixels
        Right edge of last bin defaults to maximum value of x
    Returns
    -------
    bin_centers : array
        The center of each bin in R. shape is (nx, )
    ring_averages : array
        Radial average of the image. shape is (nx, ).
    """
    radial_val = utils.radial_grid(calibrated_center, image.shape, pixel_size) 
    
    
    if mask is not None:  
        #maks = np.ones_like(  image )
        mask = np.array( mask, dtype = bool)
        binr = radial_val[mask]
        image_mask =     np.array( image )[mask]
        
    else:        
        binr = np.ravel( radial_val ) 
        image_mask = np.ravel(image) 
        
    bin_edges, sums, counts = utils.bin_1D( binr,
                                           image_mask,
                                           nx,
                                           min_x=min_x,
                                           max_x=max_x)
    
    #print (counts)
    th_mask = counts > threshold
    ring_averages = sums[th_mask] / counts[th_mask]

    bin_centers = utils.bin_edges_to_centers(bin_edges)[th_mask]

    return bin_centers, ring_averages 



def get_angular_average( avg_img, mask, pargs,   min_r, max_r, 
                        nx=3600, plot_ = False ,   save=False, *argv,**kwargs):   
    """get a angular average of an image        
    Parameters
    ----------
    
    avg_img: 2D-array, the image
    mask: 2D-array  
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength    
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)

    nx : int, optional
        number of bins in x
        defaults is 1500 bins
        
    plot_: a boolen type, if True, plot the one-D curve
    plot_qinpixel:a boolen type, if True, the x-axis of the one-D curve is q in pixel; else in real Q
    
    Returns
    -------
    ang: ang in degree
    iq: intensity of circular average
     
     
     
    """   
    
    center, Ldet, lambda_, dpix= pargs['center'],  pargs['Ldet'],  pargs['lambda_'],  pargs['dpix']
    uid = pargs['uid']
    
    angq, ang = angular_average( avg_img,  calibrated_center=center, pixel_size=(dpix,dpix), nx =nx,
                    min_r = min_r , max_r = max_r, mask=mask )
 
    
    if plot_:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111) 
        ax.plot(angq,  ang , '-o')             
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("I(ang)")
        #ax.legend(loc = 'best')  
        uid = pargs['uid']
        title = ax.set_title('Uid= %s--t~I(Ang)'%uid)        
        title.set_y(1.01)
        
        if save:
            dt =datetime.now()
            CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs['path']
            uid = pargs['uid']
 
            fp = path + 'Uid= %s--Ang-Iq~t-'%uid + CurTime + '.png'         
            fig.savefig( fp, dpi=fig.dpi)
            
        plt.show()
 
        
    return  angq, ang





def angular_average(image, calibrated_center, threshold=0, nx=1500,
                     pixel_size=(1, 1),  min_r=None, max_r=None, min_x=None, max_x=None, mask=None):
    """Angular_average of the the image data
     
    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    calibrated_center : tuple
        The center of the image in pixel units
        argument order should be (row, col)
    threshold : int, optional
        Ignore counts above `threshold`
        default is zero
    nx : int, optional
        number of bins in x
        defaults is 100 bins
    pixel_size : tuple, optional
        The size of a pixel (in a real unit, like mm).
        argument order should be (pixel_height, pixel_width)
        default is (1, 1)
        
    min_r: float, optional number of pixels
        The min r, e.g., the starting radius for angule average
    max_r:float, optional number of pixels
        The max r, e.g., the ending radius for angule average
        max_r - min_r gives the width of the angule average
        
    min_x : float, optional number of pixels
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional number of pixels
        Right edge of last bin defaults to maximum value of x
    Returns
    -------
    bin_centers : array
        The center of each bin in degree shape is (nx, )
    ring_averages : array
        Radial average of the image. shape is (nx, ).
    """
     
    angle_val = utils.angle_grid(calibrated_center, image.shape, pixel_size) 
        
    if min_r is None:
        min_r=0
    if max_r is None:
        max_r = np.sqrt(  (image.shape[0] - center[0])**2 +  (image.shape[1] - center[1])**2 )    
    r_mask = make_ring_mask( calibrated_center, image.shape, min_r, max_r )
    
    
    if mask is not None:  
        #maks = np.ones_like(  image )
        mask = np.array( mask*r_mask, dtype = bool)
        
        bina = angle_val[mask]
        image_mask =     np.array( image )[mask]
        
    else:        
        bina = np.ravel( angle_val ) 
        image_mask = np.ravel(image*r_mask) 
        
    bin_edges, sums, counts = utils.bin_1D( bina,
                                           image_mask,
                                           nx,
                                           min_x=min_x,
                                           max_x=max_x)
    
    #print (counts)
    th_mask = counts > threshold
    ang_averages = sums[th_mask] / counts[th_mask]

    bin_centers = utils.bin_edges_to_centers(bin_edges)[th_mask]

    return bin_centers*180/np.pi, ang_averages 

def get_circular_average( avg_img, mask, pargs, show_pixel=True,
                          nx=1500, plot_ = False ,   save=False, *argv,**kwargs):   
    """get a circular average of an image        
    Parameters
    ----------
    
    avg_img: 2D-array, the image
    mask: 2D-array  
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength    
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)

    nx : int, optional
        number of bins in x
        defaults is 1500 bins
        
    plot_: a boolen type, if True, plot the one-D curve
    plot_qinpixel:a boolen type, if True, the x-axis of the one-D curve is q in pixel; else in real Q
    
    Returns
    -------
    qp: q in pixel
    iq: intensity of circular average
    q: q in real unit (A-1)
     
     
    """   
    
    center, Ldet, lambda_, dpix= pargs['center'],  pargs['Ldet'],  pargs['lambda_'],  pargs['dpix']
    uid = pargs['uid']
    
    qp_, iq = circular_average(avg_img, 
        center, threshold=0, nx=nx, pixel_size=(dpix, dpix), mask=mask)

    #  convert bin_centers from r [um] to two_theta and then to q [1/px] (reciprocal space)
    two_theta = utils.radius_to_twotheta(Ldet, qp_)
    q = utils.twotheta_to_q(two_theta, lambda_)
    
    qp = qp_/dpix
    
    if plot_:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        
        if   show_pixel:
            ax2.semilogy(qp, iq, '-o')
            ax1.semilogy(q,  iq , '-o')
            
            ax2.set_xlabel('q (pixel)')             
            ax1.set_xlabel('q ('r'$\AA^{-1}$)')
             
        else:
           
            ax2.semilogy(q,  iq , '-o')
            ax1.semilogy(qp, iq, '-o')
            ax1.set_xlabel('q (pixel)')             
            ax2.set_xlabel('q ('r'$\AA^{-1}$)')
        ax2.cla()
        ax1.set_ylabel('I(q)')
        
                    
        if 'xlim' in kwargs.keys():
            ax1.set_xlim(    kwargs['xlim']  )    
            x1,x2 =  kwargs['xlim']
            w = np.where( (q >=x1 )&( q<=x2) )[0]
                        
            ax2.set_xlim(  [ qp[w[0]], qp[w[-1]]]     )    
            
        if 'ylim' in kwargs.keys():
            ax1.set_ylim(    kwargs['ylim']  )
            
            
        #ax1.semilogy(qp, np.ones_like( iq) , '-o')
        
        
        #ax2.semilogy(q, iq, '-o')
        
        #ax2.cla()
        
        title = ax2.set_title('Uid= %s--Circular Average'%uid)        
        title.set_y(1.1)
        fig.subplots_adjust(top=0.85)

        
        #if plot_qinpixel:
        #ax2.set_xlabel('q (pixel)')
        #else:
        #ax1.set_xlabel('q ('r'$\AA^{-1}$)')
        #axes.set_xlim(30,  1050)
        #axes.set_ylim(-0.0001, 10000)
        if save:
            dt =datetime.now()
            CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs['path']
            fp = path + 'Uid= %s--Circular Average'%uid + CurTime + '.png'         
            fig.savefig( fp, dpi=fig.dpi)
        
        plt.show()
        
    return  qp, iq, q


def get_distance(p1,p2):
    '''Calc the distance between two point'''
    return np.sqrt(  (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2   )


def calc_q(L,a,wv):
    ''' calc_q(L,a,wv) - calculate the q value for length L, transverse
            distance a and wavelength wv.
        Use this to calculate the speckle size
        
        L - sample to detector distance (mm)
        a - pixel size transverse length from beam direction (mm)
        wv - wavelength
        Units of L and a should match and resultant q is in inverse units of wv.
    '''
    theta = np.arctan2(a,L)
    q = 4*np.pi*np.sin(theta/2.)/wv
    return q




def get_t_iq( data_series, frame_edge, mask, pargs, nx=1500, plot_ = False , save=False, *argv,**kwargs):   
    '''Get t-dependent Iq 
    
        Parameters        
        ----------
        data_series:  a image series
        frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
        mask:  a image mask 
        
        nx : int, optional
            number of bins in x
            defaults is 1500 bins
        plot_: a boolen type, if True, plot the time~one-D curve with qp as x-axis

        Returns
        ---------
        qp: q in pixel
        iq: intensity of circular average
        q: q in real unit (A-1)
     
    '''   
       
    Nt = len( frame_edge )
    iqs = list( np.zeros( Nt ) )
    for i in range(Nt):
        t1,t2 = frame_edge[i]
        #print (t1,t2)
        avg_img = get_avg_img( data_series[t1:t2], sampling = 1,
                         plot_ = False )
        qp, iqs[i], q = get_circular_average( avg_img, mask,pargs, nx=nx,
                           plot_ = False)
        
    if plot_:
        fig,ax = plt.subplots(figsize=(8, 6))
        for i in range(  Nt ):
            t1,t2 = frame_edge[i]
            ax.semilogy(q, iqs[i], label="frame: %s--%s"%( t1,t2) )
            #ax.set_xlabel("q in pixel")
            ax.set_xlabel('Q 'r'($\AA^{-1}$)')
            ax.set_ylabel("I(q)")
            
        if 'xlim' in kwargs.keys():
            ax.set_xlim(    kwargs['xlim']  )    
        if 'ylim' in kwargs.keys():
            ax.set_ylim(    kwargs['ylim']  )
 

        ax.legend(loc = 'best')  
        
        uid = pargs['uid']
        title = ax.set_title('Uid= %s--t~I(q)'%uid)        
        title.set_y(1.01)
        if save:
            dt =datetime.now()
            CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs['path']
            uid = pargs['uid']
            fp = path + 'Uid= %s--Iq~t-'%uid + CurTime + '.png'         
            fig.savefig( fp, dpi=fig.dpi)
            
        plt.show()
        
    
    return qp, np.array( iqs ),q


    
def get_t_ang( data_series, frame_edge, mask, center, pixel_size, min_r, max_r,pargs,
                nx=1500, plot_ = False , save=False, *argv,**kwargs):   
    '''Get t-dependent angule intensity 
    
        Parameters        
        ----------
        data_series:  a image series
        frame_edge: list, the ROI frame regions, e.g., [  [0,100], [200,400] ]
        mask:  a image mask 
        
        pixel_size : tuple, optional
            The size of a pixel (in a real unit, like mm).
            argument order should be (pixel_height, pixel_width)
            default is (1, 1)
        center: the beam center in pixel 
        min_r: float, optional number of pixels
            The min r, e.g., the starting radius for angule average
        max_r:float, optional number of pixels
            The max r, e.g., the ending radius for angule average
            
            max_r - min_r gives the width of the angule average
        
        nx : int, optional
            number of bins in x
            defaults is 1500 bins
        plot_: a boolen type, if True, plot the time~one-D curve with qp as x-axis

        Returns
        ---------
        qp: q in pixel
        iq: intensity of circular average
        q: q in real unit (A-1)
     
    '''   
       
    Nt = len( frame_edge )
    iqs = list( np.zeros( Nt ) )
    for i in range(Nt):
        t1,t2 = frame_edge[i]
        #print (t1,t2)
        avg_img = get_avg_img( data_series[t1:t2], sampling = 1,
                         plot_ = False )
        qp, iqs[i] = angular_average( avg_img, center, pixel_size=pixel_size, 
                                     nx=nx, min_r=min_r, max_r = max_r, mask=mask ) 
        
    if plot_:
        fig,ax = plt.subplots(figsize=(8, 8))
        for i in range(  Nt ):
            t1,t2 = frame_edge[i]
            #ax.semilogy(qp* 180/np.pi, iqs[i], label="frame: %s--%s"%( t1,t2) )
            ax.plot(qp, iqs[i], label="frame: %s--%s"%( t1,t2) )
            ax.set_xlabel("angle (deg)")
            ax.set_ylabel("I(ang)")
        ax.legend(loc = 'best')  
        uid = pargs['uid']
        title = ax.set_title('Uid= %s--t~I(Ang)'%uid)        
        title.set_y(1.01)
        
        if save:
            dt =datetime.now()
            CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = pargs['path']
            uid = pargs['uid']
 
            fp = path + 'Uid= %s--Ang-Iq~t-'%uid + CurTime + '.png'         
            fig.savefig( fp, dpi=fig.dpi)
            
        plt.show()
        
    
    return qp, np.array( iqs ) 


def make_ring_mask(center, shape, min_r, max_r  ):
    """
    Make a ring mask.

    Parameters
    ----------
    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).        

    min_r: float, optional number of pixels
        The min r, e.g., the starting radius of the ring
    max_r:float, optional number of pixels
        The max r, e.g., the ending radius of the ring
        max_r - min_r gives the width of the ring
    Returns
    -------
    ring_mask : array


    """
    r_val = utils.radial_grid(center, shape, [1.,1.] ) 
    r_mask = np.zeros_like( r_val, dtype=np.int32)
    r_mask[np.where(  (r_val >min_r)  & (r_val < max_r ) )] = 1

    return r_mask   


def _make_roi(coords, edges, shape):
    """ Helper function to create ring rois and bar rois
    Parameters
    ----------
    coords : array
        shape is image shape
    edges : list
        List of tuples of inner (left or top) and outer (right or bottom)
        edges of each roi.
        e.g., edges=[(1, 2), (11, 12), (21, 22)]
    shape : tuple
        Shape of the image in which to create the ROIs
        e.g., shape=(512, 512)
    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are
        specified in `edges`.
        Has shape=`image shape`
    """
    label_array = np.digitize(coords, edges, right=False)
    # Even elements of label_array are in the space between rings.
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    return label_array.reshape(shape)


def angulars(edges, center, shape):
    """
    Draw annual (angluar-shaped) shaped regions of interest.
    Each ring will be labeled with an integer. Regions outside any ring will
    be filled with zeros.
    Parameters
    ----------
    edges: list
        giving the inner and outer angle in unit of radians
        e.g., [(1, 2), (11, 12), (21, 22)]
    center: tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).
    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges.
    """
    edges = np.atleast_2d(np.asarray(edges)).ravel() 
    if not 0 == len(edges) % 2:
        raise ValueError("edges should have an even number of elements, "
                         "giving inner, outer radii for each angular")
    if not np.all( np.diff(edges) > 0):         
        raise ValueError("edges are expected to be monotonically increasing, "
                         "giving inner and outer radii of each angular from "
                         "r=0 outward")

    angle_val = utils.angle_grid( center,  shape) .ravel()        
     
    return _make_roi(angle_val, edges, shape)




    

def get_angular_mask( mask,  inner_angle=0, outer_angle = 360, width = None, edges = None,
                     num_angles = 12, center = None, dpix=[1,1]    ):
     
    ''' 
    mask: 2D-array 
    inner_angle # the starting angle in unit of degree
    outer_angle #  the ending angle in unit of degree
    width       # width of each angle, in degree, default is None, there is no gap between the neighbour angle ROI
    edges: default, None. otherwise, give a customized angle edges
    num_angles    # number of angles
     
    center: the beam center in pixel    
    dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
        
    Returns
    -------
    ang_mask: a ring mask, np.array
    ang_center: ang in unit of degree
    ang_val: ang edges in degree
    
    '''
    
    #center, Ldet, lambda_, dpix= pargs['center'],  pargs['Ldet'],  pargs['lambda_'],  pargs['dpix']
    
    #spacing =  (outer_radius - inner_radius)/(num_rings-1) - 2    # spacing between rings
    #inner_angle,outer_angle = np.radians(inner_angle),  np.radians(outer_angle)
    
    #if edges is  None:
    #    ang_center =   np.linspace( inner_angle,outer_angle, num_angles )  
    #    edges = np.zeros( [ len(ang_center), 2] )
    #    if width is None:
    #        width = ( -inner_angle + outer_angle)/ float( num_angles -1 + 1e-10 )
    #    else:
    #        width =  np.radians( width )
    #    edges[:,0],edges[:,1] = ang_center - width/2, ang_center + width/2 
        

    if edges is None:
        if num_angles!=1:
            spacing =  (outer_angle - inner_angle - num_angles* width )/(num_angles-1)      # spacing between rings
        else:
            spacing = 0
        edges = roi.ring_edges(inner_angle, width, spacing, num_angles) 
        
    #print (edges)
    angs = angulars( np.radians( edges ), center, mask.shape)    
    ang_center = np.average(edges, axis=1)        
    ang_mask = angs*mask
    ang_mask = np.array(ang_mask, dtype=int)    
    
    labels, indices = roi.extract_label_indices(ang_mask)
    nopr = np.bincount( np.array(labels, dtype=int) )[1:]
   
    if len( np.where( nopr ==0 )[0] !=0):
        print (nopr)
        print ("Some angs contain zero pixels. Please redefine the edges.")      
    return ang_mask, ang_center, edges


def get_ring_mask(  mask, inner_radius=40, outer_radius = 762, width = 6, num_rings = 12, edges=None, pargs=None   ):
     
    ''' 
    mask: 2D-array 
    inner_radius #radius of the first ring
    outer_radius # radius of the last ring
    width       # width of each ring
    num_rings    # number of rings
    pargs: a dict, should contains
        center: the beam center in pixel
        Ldet: sample to detector distance
        lambda_: the wavelength    
        dpix, the pixel size in mm. For Eiger1m/4m, the size is 75 um (0.075 mm)
        
    Returns
    -------
    ring_mask: a ring mask, np.array
    q_ring_center: q in real unit (A-1)
    q_ring_val: q edges in A-1 
    
    '''
    
    center, Ldet, lambda_, dpix= pargs['center'],  pargs['Ldet'],  pargs['lambda_'],  pargs['dpix']
    
    #spacing =  (outer_radius - inner_radius)/(num_rings-1) - 2    # spacing between rings
    #qc = np.int_( np.linspace( inner_radius,outer_radius, num_rings ) )
    #edges = np.zeros( [ len(qc), 2] )
    #if width%2: 
    #   edges[:,0],edges[:,1] = qc - width//2,  qc + width//2 +1
    #else:
    #    edges[:,0],edges[:,1] = qc - width//2,  qc + width//2 
    
    #  find the edges of the required rings
    if edges is None:
        if num_rings!=1:
            spacing =  (outer_radius - inner_radius - num_rings* width )/(num_rings-1)      # spacing between rings
        else:
            spacing = 0
        edges = roi.ring_edges(inner_radius, width, spacing, num_rings)    
        
    #print ( edges )        
    
    two_theta = utils.radius_to_twotheta(Ldet, edges*dpix)
    q_ring_val = utils.twotheta_to_q(two_theta, lambda_)

    q_ring_center = np.average(q_ring_val, axis=1)    
    
    rings = roi.rings(edges, center, mask.shape)
    ring_mask = rings*mask
    ring_mask = np.array(ring_mask, dtype=int)    
    
    labels, indices = roi.extract_label_indices(ring_mask)
    nopr = np.bincount( np.array(labels, dtype=int) )[1:]
   
    if len( np.where( nopr ==0 )[0] !=0):
        print (nopr)
        print ("Some rings contain zero pixels. Please redefine the edges.")      
    return ring_mask, q_ring_center, q_ring_val

 
    


def get_ring_anglar_mask(ring_mask, ang_mask,    
                         q_ring_center, ang_center   ):
    '''get   ring_anglar mask  '''
    
    ring_max = ring_mask.max()
    
    ang_mask_ = np.zeros( ang_mask.shape  )
    ind = np.where(ang_mask!=0)
    ang_mask_[ind ] =  ang_mask[ ind ] + 1E9  #add some large number to qr
    
    ring_ang = ring_mask * ang_mask_  
    
    #convert label_array_qzr to [1,2,3,...]
    ura = np.unique( ring_ang )[1:]
    
    ur = np.unique( ring_mask )[1:]
    ua = np.unique( ang_mask )[1:]
    #print (uqzr)     
    
    ring_ang_ = np.zeros_like( ring_ang )
    newl = np.arange( 1, len(ura)+1)
    
    rc= [  [ q_ring_center[i]]*len( ua ) for i in range(len( ur ))  ]
    ac =list( ang_center) * len( ur )
    
    #rc =list( q_ring_center) * len( ua )
    #ac= [  [ ang_center[i]]*len( ur ) for i in range(len( ua ))  ]
    
    for i, label in enumerate(ura):
        #print (i, label)
        ring_ang_.ravel()[ np.where(  ring_ang.ravel() == label)[0] ] = newl[i]    
    
    
    return   np.int_(ring_ang_), np.concatenate( np.array( rc )),  np.array( ac ) 




def show_ring_ang_roi( data, rois,   alpha=0.3):    
        
    ''' 
    May 16, 2016, Y.G.@CHX
    plot a saxs image with rois( a label array) 
    
    Parameters:
        data: 2-D array, a gisaxs image 
        rois:  2-D array, a label array         
 
         
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

        x_val = int( ind.mean() )
        #print (xval, y)
        ax.text(x_val, y_val, c, va='center', ha='center')

        #print (x_val1,x_val2)

    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    #ax.set_xlabel(r'$q_r$', fontsize=22)
    #ax.set_ylabel(r'$q_z$',fontsize=22)

    plt.show()



 
    
    
    
    
def plot_qIq_with_ROI( q, iq, q_ring_center, logs=True, *argv,**kwargs):   
    '''plot q~Iq with interested q rings'''

    uid = 'uid'
    if 'uid' in kwargs.keys():
        uid = kwargs['uid'] 
        
    #Plot the circular average as a 
    fig, axes = plt.subplots( figsize=(8, 6))
    if logs:
        axes.semilogy(q, iq, '-o')
    else:        
        axes.plot(q,  iq, '-o')
        
    axes.set_title('Uid= %s--Circular Average with the Q ring values'%uid)
    axes.set_ylabel('I(q)')
    axes.set_xlabel('Q 'r'($\AA^{-1}$)')
    #axes.set_xlim(0, 0.02)
    #axes.set_xlim(-0.00001, 0.1)
    #axes.set_ylim(-0.0001, 10000)
    #axes.set_ylim(0, 100)
    
    if 'xlim' in kwargs.keys():
         axes.set_xlim(    kwargs['xlim']  )    
    if 'ylim' in kwargs.keys():
         axes.set_ylim(    kwargs['ylim']  )
 
        
        
    num_rings = len( np.unique( q_ring_center) )
    for i in range(num_rings):
        axes.axvline(q_ring_center[i] )#, linewidth = 5  )
    plt.show()


    
def get_each_ring_mean_intensity( data_series, ring_mask, sampling, timeperframe, plot_ = True , save=False, *argv,**kwargs):   
    
    """
    get time dependent mean intensity of each ring
    """
    mean_int_sets, index_list = roi.mean_intensity(np.array(data_series[::sampling]), ring_mask) 
    
    times = np.arange(len(data_series))*timeperframe  # get the time for each frame
    num_rings = len( np.unique( ring_mask)[1:] )
    if plot_:
        fig, ax = plt.subplots(figsize=(8, 8))
        uid = 'uid'
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        
        ax.set_title("Uid= %s--Mean intensity of each ring"%uid)
        for i in range(num_rings):
            ax.plot( mean_int_sets[:,i], label="Ring "+str(i+1),marker = 'o', ls='-')
            ax.set_xlabel("Time")
            ax.set_ylabel("Mean Intensity")
        ax.legend(loc = 'best') 
                
        if save:
            dt =datetime.now()
            CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)             
            path = kwargs['path']              
            fp = path + "Uid= %s--Mean intensity of each ring-"%uid + CurTime + '.png'         
            fig.savefig( fp, dpi=fig.dpi)
        
        plt.show()
    return times, mean_int_sets


 





    
def plot_saxs_g2( g2, taus, res_pargs=None, *argv,**kwargs):     
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
        q_ring_center = res_pargs[ 'q_ring_center']

    else:
        
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'

        if 'q_ring_center' in kwargs.keys():
            q_ring_center = kwargs[ 'q_ring_center']
        else:
            q_ring_center = np.arange(  g2.shape[1] )

        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
    
    num_rings = g2.shape[1]
    sx = int(round(np.sqrt(num_rings)) )
    if num_rings%sx == 0: 
        sy = int(num_rings/sx)
    else:
        sy=int(num_rings/sx+1)  
    
    #print (num_rings)
    
    if num_rings!=1:

        fig = plt.figure(figsize=(12, 10))
        plt.axis('off')
        #plt.axes(frameon=False)
    
        plt.xticks([])
        plt.yticks([])
        
        
    else:
        fig = plt.figure(figsize=(8,8))
        #print ('here')
    plt.title('uid= %s'%uid,fontsize=20, y =1.06)        
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
 
 
    dt =datetime.now()
    CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)        
                
    fp = path + 'g2--uid=%s'%(uid) + CurTime + '.png'
    fig.savefig( fp, dpi=fig.dpi)        
    fig.tight_layout()  
    plt.show()
                
 



            
    
    

def plot_saxs_two_g2( g2, taus, g2b, tausb,res_pargs=None, *argv,**kwargs):     
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
        q_ring_center = res_pargs[ 'q_ring_center']

    else:
        
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'

        if 'q_ring_center' in kwargs.keys():
            q_ring_center = kwargs[ 'q_ring_center']
        else:
            q_ring_center = np.arange(  g2.shape[1] )

        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
            


    
    num_rings = g2.shape[1]
    sx = int(round(np.sqrt(num_rings)) )
    if num_rings%sx == 0: 
        sy = int(num_rings/sx)
    else:
        sy=int(num_rings/sx+1)
        
    uid = 'uid'
    if 'uid' in kwargs.keys():
        uid = kwargs['uid']     


    sx = int(round(np.sqrt(num_rings)) )
    if num_rings%sx == 0: 
        sy = int(num_rings/sx)
    else:
        sy=int(num_rings/sx+1)
    if num_rings!=1:
        
        #fig = plt.figure(figsize=(14, 10))
        fig = plt.figure(figsize=(12, 10))
        plt.axis('off')
        #plt.axes(frameon=False)
        #print ('here')
        plt.xticks([])
        plt.yticks([])

        
    else:
        fig = plt.figure(figsize=(8,8))
        
    plt.title('uid= %s'%uid,fontsize=20, y =1.06)  
    plt.axis('off')
    for sn in range(num_rings):
        ax = fig.add_subplot(sx,sy,sn+1 )
        ax.set_ylabel("g2") 
        ax.set_title(" Q= " + '%.5f  '%(q_ring_center[sn]) + r'$\AA^{-1}$')     
        y=g2b[:, sn]
        ax.semilogx( tausb, y, '--rs', markersize=6,label= 'from_two-time')  

        y2=g2[:, sn]
        ax.semilogx(taus, y2, '--bo', markersize=6,  label= 'from_one_time') 
        if 'ylim' in kwargs:
            ax.set_ylim( kwargs['ylim'])
        elif 'vlim' in kwargs:
            vmin, vmax =kwargs['vlim']
            ax.set_ylim([min(y2)*vmin, max(y2[1:])*vmax ])
        else:
            pass
        if 'xlim' in kwargs:
            ax.set_xlim( kwargs['xlim'])
        
        if sn==0:
            ax.legend(loc = 'best')       


    dt =datetime.now()
    CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)        
    fig.tight_layout()            
    fp = path + 'g2--uid=%s'%(uid) + CurTime + '--twog2.png'
    fig.savefig( fp, dpi=fig.dpi)        
     
    plt.show()
            
    
    

    
#plot g2 results

def plot_saxs_rad_ang_g2( g2, taus, res_pargs=None, *argv,**kwargs):     
    '''plot g2 results of segments with radius and angle partation , 
    
       g2: one-time correlation function
       taus: the time delays  
       res_pargs, a dict, can contains
           uid/path/qr_center/qz_center/
       kwargs: can contains
           vlim: [vmin,vmax]: for the plot limit of y, the y-limit will be [vmin * min(y), vmx*max(y)]
           ylim/xlim: the limit of y and x
       
       e.g.
       plot_saxs_rad_ang_g2( g2b, taus= np.arange( g2b.shape[0]) *timeperframe, q_ring_center = q_ring_center, ang_center=ang_center, vlim=[.99, 1.01] )
           
      ''' 

    
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path'] 
        q_ring_center= res_pargs[ 'q_ring_center']
        num_qr = len( q_ring_center)
        ang_center = res_pargs[ 'ang_center']
        num_ang = len( ang_center )
    
    else:
    
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'
        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
        if 'q_ring_center' in kwargs.keys():
            q_ring_center = kwargs[ 'q_ring_center']
            num_qr = len( q_ring_center)
        else:
            print( 'Please give q_ring_center')
        if 'ang_center' in kwargs.keys():
            ang_center = kwargs[ 'ang_center']
            num_qa = len( ang_center)
        else:
            print( 'Please give ang_center') 


        

    for qr_ind in range(num_qr):
        fig = plt.figure(figsize=(10, 12))
        #fig = plt.figure()
        title_qr = ' Qr= %.5f  '%( q_ring_center[qr_ind]) + r'$\AA^{-1}$' 
        plt.title('uid= %s:--->'%uid + title_qr,fontsize=20, y =1.1) 
        #print (qz_ind,title_qz)
        if num_qr!=1:plt.axis('off')
        sx = int(round(np.sqrt(num_qa)) )
        if num_qa%sx == 0: 
            sy = int(num_qa/sx)
        else: 
            sy=int(num_qa/sx+1) 
        for sn in range(num_qa):
            ax = fig.add_subplot(sx,sy,sn+1 )
            ax.set_ylabel("g2") 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16) 
            i = sn + qr_ind * num_qa
            title_qa = " Angle= " + '%.2f'%( ang_center[sn]) + r'$^\circ$' + '( %d )'%i
            if num_qr==1:

                title = 'uid= %s:--->'%uid + title_qr + '__' +  title_qa
            else:
                title = title_qa
            ax.set_title( title  )


            y=g2[:, i]
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
                
        fp = path + 'g2--uid=%s-qr=%s'%(uid,q_ring_center[qr_ind]) + CurTime + '.png'
        fig.savefig( fp, dpi=fig.dpi)        
        fig.tight_layout()  
        plt.show()

        
        
 

def fit_saxs_rad_ang_g2( g2, taus, res_pargs=None,function='simple_exponential', *argv,**kwargs):    
     
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
    
    
    
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path'] 
        q_ring_center= res_pargs[ 'q_ring_center']
        num_qr = len( q_ring_center)
        ang_center = res_pargs[ 'ang_center']
        num_ang = len( ang_center )
    
    else:
    
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'
        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
        if 'q_ring_center' in kwargs.keys():
            q_ring_center = kwargs[ 'q_ring_center']
            num_qr = len( q_ring_center)
        else:
            print( 'Please give q_ring_center')
        if 'ang_center' in kwargs.keys():
            ang_center = kwargs[ 'ang_center']
            num_qa = len( ang_center)
        else:
            print( 'Please give ang_center') 

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
    
        

    for qr_ind in range(num_qr):
        fig = plt.figure(figsize=(10, 12))
        #fig = plt.figure()
        title_qr = ' Qr= %.5f  '%( q_ring_center[qr_ind]) + r'$\AA^{-1}$' 
        plt.title('uid= %s:--->'%uid + title_qr,fontsize=20, y =1.1) 
        #print (qz_ind,title_qz)
        if num_qr!=1:plt.axis('off')
        sx = int(round(np.sqrt(num_qa)) )
        if num_qa%sx == 0: 
            sy = int(num_qa/sx)
        else: 
            sy=int(num_qa/sx+1) 
        for sn in range(num_qa):
            ax = fig.add_subplot(sx,sy,sn+1 )
            ax.set_ylabel("g2") 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16) 
            i =  sn + qr_ind * num_qa
            title_qa = " Angle= " + '%.2f'%( ang_center[sn]) + r'$^\circ$' + '( %d )'%i
            if num_qr==1:

                title = 'uid= %s:--->'%uid + title_qr + '__' +  title_qa
            else:
                title = title_qa
            ax.set_title( title  )

            
            y=g2[1:, i]
            #print (y.shape)
            
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
            
            
            txts = r'$\gamma$' + r'$ = %.3f$'%(1/rate[i]) +  r'$ s$'
            ax.text(x =0.02, y=.55, s=txts, fontsize=14, transform=ax.transAxes)     
            txts = r'$\alpha$' + r'$ = %.3f$'%(alpha[i])  
            #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
            ax.text(x =0.02, y=.45, s=txts, fontsize=14, transform=ax.transAxes)  

            txts = r'$baseline$' + r'$ = %.3f$'%( baseline[i]) 
            ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)  
        
        
        
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
                
        fp = path + 'g2--uid=%s-qr=%s'%(uid,q_ring_center[qr_ind]) + CurTime + '.png'
        fig.savefig( fp, dpi=fig.dpi)        
        fig.tight_layout()  
        plt.show()
        
    #fig.tight_layout()  
    #plt.show()
    result = dict( beta=beta, rate=rate, alpha=alpha, baseline=baseline )
    
    return result
                
    
    

def save_saxs_g2(  g2, res_pargs, taus=None, filename=None ):
    
    '''save g2 results, 
       res_pargs should contain
           g2: one-time correlation function
           res_pargs: contions taus, q_ring_center values
           path:
           uid:
      
      '''
    
    
    if taus is None:
        taus = res_pargs[ 'taus']
    q_ring_center = res_pargs['q_ring_center']
    path = res_pargs['path']
    uid = res_pargs['uid']
    
    df = DataFrame(     np.hstack( [ (taus).reshape( len(g2),1) ,  g2] )  ) 
    df.columns = (   ['tau'] + [str(qs) for qs in q_ring_center ]    )
    
    dt =datetime.now()
    CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)  
    
    if filename is None:
        filename = 'g2'
    filename += '-uid=%s-%s.csv' % (uid,CurTime)    
    filename1 = os.path.join(path, filename)
    df.to_csv(filename1)
    print( 'The g2 is saved in %s with filename as %s'%( path, filename))

    
    
    

    
def stretched_auto_corr_scat_factor(lags, beta, relaxation_rate, alpha=1.0, baseline=1):
    beta=abs(beta)
    relaxation_rate = abs( relaxation_rate )
    baseline = abs( baseline )
    return beta * (np.exp(-2 * relaxation_rate * lags))**alpha + baseline

def simple_exponential(lags, beta, relaxation_rate,  baseline=1):
    beta=abs(beta)
    relaxation_rate = abs( relaxation_rate )
    baseline = abs( baseline )
    return beta * np.exp(-2 * relaxation_rate * lags) + baseline


def fit_saxs_g2( g2, res_pargs=None, function='simple_exponential', *argv,**kwargs):    
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
    
            
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path'] 
        taus = res_pargs[ 'taus']
        q_ring_center = res_pargs[ 'q_ring_center']

    else:
        taus =  kwargs['taus'] 
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'

        if 'q_ring_center' in kwargs.keys():
            q_ring_center = kwargs[ 'q_ring_center']
        else:
            q_ring_center = np.arange(  g2.shape[1] )

        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
        
    if 'fit_range' in kwargs.keys():
        fit_range = kwargs['fit_range'] 
    else:
        fit_range=None
 
    
    num_rings = g2.shape[1]
    

    beta = np.zeros(   num_rings )  #  contrast factor
    rate = np.zeros(   num_rings )  #  relaxation rate
    alpha = np.zeros(   num_rings )  #  alpha
    baseline = np.zeros(   num_rings )  #  baseline

    sx = int( round (np.sqrt(num_rings)) )
    if num_rings%sx==0:
        sy = int(num_rings/sx)
    else:
        sy = int(num_rings/sx+1)
        
    if function=='simple_exponential' or function=='simple':
        mod = Model(stretched_auto_corr_scat_factor,  independent_vars=['alpha', 'lags'])   
        
    elif function=='stretched_exponential' or function=='stretched':
        mod = Model(stretched_auto_corr_scat_factor,  independent_vars=[ 'lags'])   
    else:
        print ("The %s is not supported.The supported functions include simple_exponential and stretched_exponential"%function)
    
    if num_rings!=1:
        
        #fig = plt.figure(figsize=(14, 10))
        fig = plt.figure(figsize=(12, 10))
        plt.axis('off')
        #plt.axes(frameon=False)
        #print ('here')
        plt.xticks([])
        plt.yticks([])

        
    else:
        fig = plt.figure(figsize=(8,8))
        
    #fig = plt.figure(figsize=(8,8))
    plt.title('uid= %s'%uid,fontsize=20, y =1.06)   
 
    for i in range(num_rings):
        ax = fig.add_subplot(sx, sy, i+1 )
        if fit_range is not None:
            y=g2[1:, i][fit_range[0]:fit_range[1]]
            lags=taus[1:][fit_range[0]:fit_range[1]]
        else:
            y=g2[1:, i]
            lags=taus[1:]

        result1 = mod.fit(y, lags=lags, beta=.05, alpha=1.0,
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

        ax.semilogx(taus[1:], g2[1:, i], 'bo')
        
        ax.semilogx(lags, result1.best_fit, '-r')
        
        ax.set_title(" Q= " + '%.5f  '%(q_ring_center[i]) + r'$\AA^{-1}$')  
        #ax.set_ylim([min(y)*.95, max(y[1:]) *1.05])
        #ax.set_ylim([0.9999, max(y[1:]) *1.002])
        txts = r'$\gamma$' + r'$ = %.3f$'%(1/rate[i]) +  r'$ s$'
        ax.text(x =0.02, y=.55, s=txts, fontsize=14, transform=ax.transAxes)     
        txts = r'$\alpha$' + r'$ = %.3f$'%(alpha[i])  
        #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
        ax.text(x =0.02, y=.45, s=txts, fontsize=14, transform=ax.transAxes)  

        txts = r'$baseline$' + r'$ = %.3f$'%( baseline[i]) 
        ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes) 
        
        txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) #+  r'$ s^{-1}$'    
        ax.text(x =0.02, y=.25, s=txts, fontsize=14, transform=ax.transAxes) 
    
        if 'ylim' in kwargs:
            ax.set_ylim( kwargs['ylim'])
        elif 'vlim' in kwargs:
            vmin, vmax =kwargs['vlim']
            ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
        else:
            ax.set_ylim([min(y)*.95, max(y[1:]) *1.05])
        if 'xlim' in kwargs:
            ax.set_xlim( kwargs['xlim'])
            

    dt =datetime.now()
    CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)        
         
    fp = path + 'g2--uid=%s'%(uid) + CurTime + '--Fit.png'
    fig.savefig( fp, dpi=fig.dpi)        
    
    fig.tight_layout()       
    plt.show()

    result = dict( beta=beta, rate=rate, alpha=alpha, baseline=baseline )
    
    return result


def linear_fit( x,y):
    D0 = np.polyfit(x, y, 1)
    gmfit = np.poly1d(D0)    
    return D0, gmfit


def fit_q2_rate( q2, rate, plot_=True, *argv,**kwargs): 
    x= q2
    y=rate
        
    if 'fit_range' in kwargs.keys():
        fit_range = kwargs['fit_range']         
    else:    
        fit_range= None

    if 'uid' in kwargs.keys():
        uid = kwargs['uid'] 
    else:
        uid = 'uid' 

    if 'path' in kwargs.keys():
        path = kwargs['path'] 
    else:
        path = ''    

    if fit_range is not None:
        y=rate[fit_range[0]:fit_range[1]]
        x=q2[fit_range[0]:fit_range[1]] 
    
    D0, gmfit = linear_fit( x,y)
    print ('The fitted diffusion coefficient D0 is:  %.2E   A^2S-1'%D0[0])
    if plot_:
        fig,ax = plt.subplots()
        plt.title('Q2-Rate--uid= %s_Fit'%uid,fontsize=20, y =1.06)   
        
        ax.plot(q2,rate, 'ro', ls='')
        ax.plot(x,  gmfit(x),  ls='-')
        
        txts = r'$D0: %.2f$'%D0[0] + r' $A^2$' + r'$s^{-1}$'
        ax.text(x =0.15, y=.75, s=txts, fontsize=14, transform=ax.transAxes) 
        
        
        
        ax.set_ylabel('Relaxation rate 'r'$\gamma$'"($s^{-1}$)")
        ax.set_xlabel("$q^2$"r'($\AA^{-2}$)')
              
        dt =datetime.now()
        CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)     
        fp = path + 'Q2-Rate--uid=%s'%(uid) + CurTime + '--Fit.png'
        fig.savefig( fp, dpi=fig.dpi)    
    
        fig.tight_layout() 
        plt.show()
        
    return D0



def plot_gamma():
    '''not work'''
    fig, ax = plt.subplots()
    ax.set_title('Uid= %s--Beta'%uid)
    ax.set_title('Uid= %s--Gamma'%uid)
    #ax.plot(  q_ring_center**2 , 1/rate, 'ro', ls='--')

    ax.loglog(  q_ring_center , 1/result['rate'], 'ro', ls='--')
    #ax.set_ylabel('Log( Beta0 'r'$\beta$'"($s^{-1}$)")
    ax.set_ylabel('Log( Gamma )')
    ax.set_xlabel("$Log(q)$"r'($\AA^{-1}$)')
    plt.show()
