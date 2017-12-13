def bin_1D(x, y, nx=None, min_x=None, max_x=None):
    """
    Bin the values in y based on their x-coordinates

    Parameters
    ----------
    x : array
        position
    y : array
        intensity
    nx : integer, optional
        number of bins to use defaults to default bin value
    min_x : float, optional
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional
        Right edge of last bin defaults to maximum value of x

    Returns
    -------
    edges : array
        edges of bins, length nx + 1

    val : array
        sum of values in each bin, length nx

    count : array
        The number of counts in each bin, length nx
    """
    # handle default values
    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)
    if nx is None:
        nx = int(max_x - min_x) 
    # use a weighted histogram to get the bin sum
    bins = np.linspace(start=min_x, stop=max_x, num=nx+1, endpoint=True)
    val, _ = np.histogram(a=x, bins=bins, weights=y)
    # use an un-weighted histogram to get the counts
    count, _ = np.histogram(a=x, bins=bins)
    # return the three arrays
    return bins, val, count



def circular_average(image, calibrated_center, threshold=0, nx=None,
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
        mask = np.array( mask, dtype = bool)
        binr = radial_val[mask]
        image_mask =     np.array( image )[mask]        
    else:        
        binr = np.ravel( radial_val ) 
        image_mask = np.ravel(image) 
    binr_ =   binr /(np.sqrt(pixel_size[1]*pixel_size[0] ))
    bin_edges, sums, counts = bin_1D(      binr_,
                                           image_mask,
                                           nx=nx,
                                     
                                           min_x=min_x,
                                           max_x=max_x) 
    th_mask = counts > threshold
    ring_averages = sums[th_mask] / counts[th_mask]
    bin_centers = utils.bin_edges_to_centers(bin_edges)[th_mask]
    return bin_centers, ring_averages 

 
    
    
    
    
    
    
    
    
    
    
    
def seek_bad_pixels(FD, uid, badlist, filename='/XF11ID/analysis/Commissioning/badpixel.csv' ):
    
    try:
        mydict = read_dict_csv( filename )       
    except:
        mydict = {}
    bp = {}
    bp[uid] = [] #np.zeros( [len( badlist),2] )
    #i=0
    for n in badlist:
        if n>= FD.beg and n<=FD.end:
            f = FD.rdframe(n)
            w = np.where(  f == f.max())
            if len(w[0])==1:
                bp[uid].append(  [ w[0], w[1] ]  )
        #i +=1
        
    if uid not in list( mydict.keys() ): 
        save_dict_csv( bp, filename, mode='a')
        print( 'save bad pixels to %s'%filename)
    return bp    