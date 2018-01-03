def xsvsc(FD_sets, label_array, number_of_img, only_first_level= True, timebin_num=2, time_bin=None,
         max_cts=None, bad_images = None, threshold=None, imgsum=None, norm=None):   
    """
    This function will provide the probability density of detecting photons
    for different integration times.
    The experimental probability density P(K) of detecting photons K is
    obtained by histogramming the speckle counts over an ensemble of
    equivalent pixels and over a number of speckle patterns recorded
    with the same integration time T under the same condition.
    Parameters
    ----------
    image_iterable : FD, a compressed eiger file by Multifile class
         
    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).
    number_of_img : int
        number of images (how far to go with integration times when finding
        the time_bin, using skxray.utils.geometric function)
    timebin_num : int, optional
        integration time; default is 2
    max_cts : int, optional
       the brightest pixel in any ROI in any image in the image set.
       defaults to using skxray.core.roi.roi_max_counts to determine
       the brightest pixel in any of the ROIs
       
       
    bad_images: array, optional
        the bad images number list, the XSVS will not analyze the binning image groups which involve any bad images
    threshold: float, optional
        If one image involves a pixel with intensity above threshold, such image will be considered as a bad image.
    
    
    Returns
    -------
    prob_k_all : array
        probability density of detecting photons
    prob_k_std_dev : array
        standard deviation of probability density of detecting photons
    Notes
    -----
    These implementation is based on following references
    References: text [1]_, text [2]_
    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.
    .. [2] R. Bandyopadhyay, A. S. Gittings, S. S. Suh, P.K. Dixon and
       D.J. Durian "Speckle-visibilty Spectroscopy: A tool to study
       time-varying dynamics" Rev. Sci. Instrum. vol 76, p  093110, 2005.
    There is an example in https://github.com/scikit-xray/scikit-xray-examples
    It will demonstrate the use of these functions in this module for
    experimental data.
    """
    if max_cts is None:
        max_cts = roi.roi_max_counts(FD_sets, label_array)
    # find the label's and pixel indices for ROI's
    labels, indices = roi.extract_label_indices(label_array)        
    nopixels = len(indices)
    # number of ROI's
    u_labels = list(np.unique(labels))
    num_roi = len(u_labels)    
    # create integration times
    if time_bin is None:time_bin = geometric_series(timebin_num, number_of_img)
    if only_first_level:
        time_bin = [1]
    # number of times in the time bin
    num_times = len(time_bin)
    # number of pixels per ROI
    num_pixels = np.bincount(labels, minlength=(num_roi+1))[1:]
    # probability density of detecting photons
    prob_k_all = np.zeros([num_times, num_roi], dtype=np.object)
    # square of probability density of detecting photons
    prob_k_pow_all = np.zeros_like(prob_k_all)
    # standard deviation of probability density of detecting photons
    prob_k_std_dev = np.zeros_like(prob_k_all)
    # get the bin edges for each time bin for each ROI
    bin_edges = np.zeros(prob_k_all.shape[0], dtype=prob_k_all.dtype)
    for i in range(num_times):
        bin_edges[i] = np.arange(max_cts*2**i)
    #start_time = time.time()  # used to log the computation time (optionally) 
    bad_frame_list = bad_images
    if bad_frame_list is None:
        bad_frame_list=[] 
    pixelist = indices    
    fra_pix = np.zeros_like( pixelist, dtype=np.float64)   
    
    for n, FD in enumerate(FD_sets): 
        timg = np.zeros(    FD.md['ncols'] * FD.md['nrows']   , dtype=np.int32   ) 
        timg[pixelist] =   np.arange( 1, len(pixelist) + 1  )        
        #print( n, FD, num_times, timebin_num, nopixels )        
        buf =  np.zeros([num_times, timebin_num, nopixels])  
        # to track processing each time level
        track_level = np.zeros( num_times )
        track_bad_level = np.zeros( num_times )
        # to increment buffer        
        cur = np.int_( np.full(num_times, timebin_num) )
        #cur =  np.full(num_times, timebin_num)         
        # to track how many images processed in each level
        img_per_level = np.zeros(num_times, dtype=np.int64)
        prob_k = np.zeros_like(prob_k_all)
        prob_k_pow = np.zeros_like(prob_k_all) 
        noframes=  FD.end - FD.beg  +1 
        for  i in tqdm(range( FD.beg , FD.end )):
            # Ring buffer, a buffer with periodic boundary conditions.
            # Images must be keep for up to maximum delay in buf.
            #buf = np.zeros([num_times, timebin_num], dtype=np.object)  # matrix of buffers
            if i in bad_frame_list:
                fra_pix[:]= np.nan
                #print( 'here is a bad frmae--%i'%i )
            else:
                fra_pix[:]=0
                (p,v) = FD.rdrawframe(i)
                w = np.where( timg[p] )[0]
                pxlist = timg[  p[w]   ] -1 

                if imgsum is None:
                    if norm is None:
                        fra_pix[ pxlist] = v[w] 
                    else: 
                        fra_pix[ pxlist] = v[w]/ norm[pxlist]   #-1.0 
                else:
                    if norm is None:
                        fra_pix[ pxlist] = v[w] / imgsum[i] 
                    else:
                        fra_pix[ pxlist] = v[w]/ imgsum[i]/  norm[pxlist] 
            #level =0            
            cur[0] = 1 + cur[0]% timebin_num
            # read each frame
            # Put the image into the ring buffer.
            img_ = fra_pix 
            # Put the ROI pixels into the ring buffer.                
            #fra_pix[:]=0        
            if threshold is not None:
                if img_.max() >= threshold:
                    print ('bad image: %s here!'%n  )   
                    img_[:]= np.nan                    
            buf[0, cur[0] - 1] = img_
            _process(num_roi, 0, cur[0] - 1, buf, img_per_level, labels,
                     max_cts, bin_edges[0], prob_k, prob_k_pow,track_bad_level)            
            # check whether the number of levels is one, otherwise
            # continue processing the next level
            level = 1
            if number_of_img>1:
                processing=1   
            else:
                processing=0
            if only_first_level:
                processing = 0
            while processing:
                #print( 'here')
                if track_level[level]:
                    
                    prev = 1 + (cur[level - 1] - 2) % timebin_num
                    cur[level] = 1 + cur[level] % timebin_num
                    
                    bufa = buf[level-1,prev-1]
                    bufb=  buf[level-1,cur[level-1]-1]
                    
                    buf[level, cur[level]-1] =  bufa + bufb  
                    
                    #print(  buf[level, cur[level]-1]  )
                    
                    track_level[level] = 0

                    _process(num_roi, level, cur[level]-1, buf, img_per_level,
                             labels, max_cts, bin_edges[level], prob_k,
                             prob_k_pow,track_bad_level)
                    level += 1
                    if level < num_times:
                        processing = 1
                    else:
                        processing = 0                    
                else:
                    track_level[level] = 1
                    processing = 0
            prob_k_all += (prob_k - prob_k_all)/(n + 1)
            prob_k_pow_all += (prob_k_pow - prob_k_pow_all)/(n + 1)

    prob_k_std_dev = np.power((prob_k_pow_all -
                               np.power(prob_k_all, 2)), .5)   
 
    for i in range(num_times):
        if   isinstance(prob_k_all[i,0], float ):
            for j in range( len(u_labels)):
                prob_k_all[i,j] = np.array(  [0] * (len(bin_edges[i]) -1 ) )
                prob_k_std_dev[i,j] = np.array(  [0] * (len(bin_edges[i]) -1 ) )
    
    return prob_k_all, prob_k_std_dev


    
    
    
    
    
    
    
    