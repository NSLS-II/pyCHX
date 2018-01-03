from pyCHX.chx_packages import *

#from pyCHX.chx_generic_functions import get_short_long_labels_from_qval_dict
#RUN_GUI = False
#from pyCHX.chx_libs import markers

####################################################################################################
##compress multi uids, sequential compress for uids, but for each uid, can apply parallel compress##
#################################################################################################
def compress_multi_uids( uids, mask, mask_dict = None, force_compress=False,  para_compress= True, bin_frame_number=1 ):
    ''' Compress time series data for a set of uids
    Parameters:
        uids: list, a list of uid
        mask: bool array, mask array
        force_compress: default is False, just load the compresssed data;
                    if True, will compress it to overwrite the old compressed data
        para_compress: apply the parallel compress algorithm
        bin_frame_number: 
    Return:
        None, save the compressed data in, by default, /XF11ID/analysis/Compressed_Data with filename as
              '/uid_%s.cmp' uid is the full uid string
    
    e.g.,  compress_multi_uids( uids, mask, force_compress= False,  bin_frame_number=1 )
    
    '''
    for uid in uids:
        print('UID: %s is in processing...'%uid)
        md = get_meta_data( uid )
        if bin_frame_number==1:
            filename = '/XF11ID/analysis/Compressed_Data' +'/uid_%s.cmp'%md['uid']
        else:
            filename = '/XF11ID/analysis/Compressed_Data' +'/uid_%s_bined--%s.cmp'%(md['uid'],bin_frame_number)
        
        imgs = load_data( uid, md['detector'], reverse= True  )
        print( imgs )
        if mask_dict is not None:
            mask = mask_dict[md['detector']]
            print('The detecotr is: %s'% md['detector'])
            
        md.update( imgs.md )
        mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(imgs, mask, md, filename, 
             force_compress= force_compress,  para_compress= para_compress,  bad_pixel_threshold= 1e14,
                            bins=bin_frame_number, num_sub= 100, num_max_para_process= 500, with_pickle=True  )
    print('Done!')
    

####################################################################################################
##get_two_time_mulit_uids, sequential cal for uids, but apply parallel for each uid ##
#################################################################################################

def get_two_time_mulit_uids( uids, roi_mask,  norm= None, bin_frame_number=1, path=None, force_generate=False ): 
    
    ''' Calculate two time correlation by using auto_two_Arrayc func for a set of uids, 
        if the two-time resutls are already created, by default (force_generate=False), just pass
    Parameters:
        uids: list, a list of uid
        roi_mask: bool array, roi mask array
        norm: the normalization array         
        path: string, where to save the two time   
        force_generate: default, False, if the two-time resutls are already created, just pass
                        if True, will force to calculate two-time no matter exist or not
        
    Return:
        None, save the two-time in as  path + uid + 'uid=%s_g12b'%uid 
        
    e.g.,
        get_two_time_mulit_uids( guids, roi_mask,  norm= norm,bin_frame_number=1, 
                        path= data_dir,force_generate=False )
    
    '''
        
    qind, pixelist = roi.extract_label_indices(roi_mask)
    for uid in uids:
        print('UID: %s is in processing...'%uid)
        md = get_meta_data( uid )        
        imgs = load_data( uid, md['detector'], reverse= True  )
        N = len(imgs)
        #print( N )
        if bin_frame_number==1:
            filename = '/XF11ID/analysis/Compressed_Data' +'/uid_%s.cmp'%md['uid']
        else:
            filename = '/XF11ID/analysis/Compressed_Data' +'/uid_%s_bined--%s.cmp'%(md['uid'],bin_frame_number)
        
        FD = Multifile(filename, 0, N//bin_frame_number)
        #print( FD.beg, FD.end)
        
        os.makedirs(path + uid + '/', exist_ok=True)
        filename =  path + uid + '/' + 'uid=%s_g12b'%uid
        doit = True
        if not force_generate:
            if os.path.exists( filename + '.npy'):
                doit=False
                print('The two time correlation function for uid=%s is already calculated. Just pass...'%uid)
        if doit:        
            data_pixel =   Get_Pixel_Arrayc( FD, pixelist,  norm= norm ).get_data()
            g12b = auto_two_Arrayc(  data_pixel,  roi_mask, index = None   )
            np.save(  filename, g12b)
            del g12b
            print( 'The two time correlation function for uid={} is saved as {}.'.format(uid, filename ))
                  
        


        

def get_series_g2_from_g12( g12b, fra_num_by_dose = None, dose_label = None,
                           good_start=0, log_taus = True, num_bufs=8, time_step=1  ):
    '''
    Get a series of one-time function from two-time by giving noframes
    Parameters:
        g12b: a two time function
        good_start: the start frame number
        fra_num_by_dose: a list, correlation number starting from index 0,
                if this number is larger than g12b length, will give a warning message, and
                will use g12b length to replace this number
                by default is None, will = [ g12b.shape[0] ] 
        dose_label: the label of each dose, also is the keys of returned g2, lag
        log_taus: if true, will only return a g2 with the correponding tau values 
                    as calculated by multi-tau defined taus
    Return:
        
        g2_series, a dict, with keys as dose_label (corrected on if warning message is given)
        lag_steps, the corresponding lags
        
    '''
    g2={}
    lag_steps = {}
    L,L,qs= g12b.shape
    if fra_num_by_dose is None:
        fra_num_by_dose  = [L]
    if dose_label is None:
        dose_label = fra_num_by_dose    
    fra_num_by_dose = sorted( fra_num_by_dose )
    dose_label = sorted( dose_label )
    for i, good_end in enumerate(fra_num_by_dose):
        key = round(dose_label[i] ,3) 
        #print( good_end )
        if good_end>L:
            warnings.warn("Warning: the dose value is too large, and please" 
                          "check the maxium dose in this data set and give a smaller dose value."
                          "We will use the maxium dose of the data.") 
            good_end = L            
        if not log_taus:            
            g2[ key ] = get_one_time_from_two_time(g12b[good_start:good_end,good_start:good_end,:] )
        else:      
            #print(  good_end,  num_bufs )
            lag_step = get_multi_tau_lag_steps(good_end,  num_bufs)
            lag_step = lag_step[ lag_step < good_end - good_start]            
            #print( len(lag_steps ) )
            lag_steps[key] = lag_step * time_step
            g2[key] = get_one_time_from_two_time(g12b[good_start:good_end,good_start:good_end,:] )[lag_step]
            
    return lag_steps, g2
                                           

    
    
def get_series_one_time_mulit_uids( uids,  qval_dict,  good_start=0,  path=None, 
                                exposure_dose = None, dead_time = 2, 
                                   num_bufs =8, save_g2=True,   ):
    ''' Calculate a dose depedent series of one time correlations from two time 
    Parameters:
        uids: list, a list of uid
        exposure_dose: list, a list x-ray exposure dose;
                by default is None, namely,  = [ max_frame_number ], 
                can be [3.34  334, 3340] in unit of ms, 
        path: string, where to load the two time, if None, ask for it
                    the real g12 path is two_time_path + uid + '/'
        qval_dict: the dictionary for q values
    Return:
        taus_uids, with keys as uid, and
                taus_uids[uid] is also a dict, with keys as dose_frame
        g2_uids, with keys as uid, and 
                g2_uids[uid] is also a dict, with keys as  dose_frame
        will also save g2 results to the 'path'
    '''
    
    if path is None:
        print( 'Please calculate two time function first by using get_two_time_mulit_uids function.')
    else:
        taus_uids = {}
        g2_uids = {}
        for i, uid in enumerate(uids):
            print('UID: %s is in processing...'%uid)
            md = get_meta_data( uid )
            imgs = load_data( uid, md['detector'], reverse= True  )
            N = len(imgs)
            if exposure_dose is None:
                exposure_dose = [N]
            g2_path =  path + uid + '/'
            g12b = np.load( g2_path + 'uid=%s_g12b.npy'%uid)
            try:
                exp_time = float( md['cam_acquire_time']) #*1000 #from second to ms
            except:                
                exp_time = float( md['exposure time']) #* 1000  #from second to ms
                
            fra_num_by_dose = get_fra_num_by_dose(  exp_dose = exposure_dose,
                                   exp_time =exp_time, dead_time = dead_time)
            
            #print( fra_num_by_dose,exposure_dose,  exp_time ) 
            
            taus_uid, g2_uid = get_series_g2_from_g12( g12b, fra_num_by_dose=fra_num_by_dose, 
                                    dose_label = exposure_dose,
                                            good_start=good_start,  num_bufs=num_bufs,
                                                     time_step = md['cam_acquire_period'] )
            g2_uids['uid_%03d=%s'%(i,uid)] = g2_uid             
            taus_uids['uid_%03d=%s'%(i,uid)] =  taus_uid  
            if save_g2:
                for k in list( g2_uid.keys()):
                    #print(k)
                    uid_ = uid + '_fra_%s_%s'%(good_start, k )
                    save_g2_general( g2_uid[k], taus=taus_uid[k],qr=np.array( list( qval_dict.values() ) )[:,0],
                             uid=uid_+'_g2.csv', path= g2_path, return_res=False )
        return taus_uids, g2_uids
     
 
    
    
def plot_dose_g2( taus_uids, g2_uids, qval_dict, ylim=[0.95, 1.05], vshift=0.1,
                   fit_res= None,  geometry= 'saxs',filename= 'dose'+'_g2', 
            path= None, function= None,  g2_labels=None, ylabel= 'g2_dose', append_name=  '_dose' ):
    '''Plot a does-dependent g2 
    taus_uids, dict, with format as {uid1: { dose1: tau_1, dose2: tau_2...}, uid2: ...}
    g2_uids, dict, with format as {uid1: { dose1: g2_1, dose2: g2_2...}, uid2: ...}
    qval_dict: a dict of qvals
    vshift: float, vertical shift value of different dose of g2
    
    '''
    
    uids = sorted( list( taus_uids.keys() ) )
    #print( uids )
    dose = sorted( list( taus_uids[ uids[0] ].keys() ) )
    g2_dict= {}
    taus_dict = {}
    if g2_labels is None:
        g2_labels = []        
    for i in range(   len( dose )):
        g2_dict[i + 1] = []
        taus_dict[i +1 ] = []
        #print ( i )
        for j in range( len( uids )):
            #print( uids[i] , dose[j])
            g2_dict[i +1 ].append(   g2_uids[ uids[j] ][  dose[i] ]  +  vshift*i  )
            taus_dict[i +1 ].append(   taus_uids[ uids[j] ][ dose[i]  ]   )
            if j ==0:
                g2_labels.append( 'Dose_%s'%dose[i] )
    #print( g2_labels)
    if True: #False:
        plot_g2_general( g2_dict, taus_dict,
                ylim=[ylim[0], ylim[1] + vshift * len(dose)],
                qval_dict = qval_dict, fit_res= None,  geometry=  geometry,filename= filename, 
        path= path, function= function,  ylabel= ylabel, g2_labels=g2_labels, append_name=  append_name )
    
    #return taus_dict, g2_dict
    
 


def run_xpcs_xsvs_single( uid, run_pargs, return_res=False):
    '''Y.G. Dec 22, 2016
       Run XPCS XSVS analysis for a single uid
       Parameters:
           uid: unique id
           run_pargs: dict, control run type and setup parameters, such as q range et.al.
       Return:
       save analysis result to csv/png/h5 files
       return_res: if true, return a dict, containing g2,g4,g12,contrast et.al. depending on the run type
       An example for the run_pargs:
       
    run_pargs=  dict(   
                    scat_geometry = 'gi_saxs'  #suport 'saxs', 'gi_saxs', 'ang_saxs' (for anisotropics saxs or flow-xpcs)
                    force_compress =  True,#False, 
                    para_compress = True,
                    run_fit_form = False,
                    run_waterfall = True,#False,
                    run_t_ROI_Inten = True,
                    #run_fit_g2 = True,
                    fit_g2_func = 'stretched',
                    run_one_time = True,#False,
                    run_two_time = True,#False,
                    run_four_time = False,
                    run_xsvs=True,
                    att_pdf_report = True,
                    show_plot = False,

                    CYCLE = '2016_3', 
                    mask_path = '/XF11ID/analysis/2016_3/masks/',
                    mask_name = 'Nov28_4M_SAXS_mask.npy', 
                    good_start   = 5, 

                    uniformq = True,
                    inner_radius= 0.005, #0.005 for 50 nm, 0.006, #for 10nm/coralpor
                    outer_radius = 0.04, #0.04 for 50 nm, 0.05, #for 10nm/coralpor 
                    num_rings = 12,
                    gap_ring_number = 6, 
                    number_rings= 1,    
                    #qcenters = [ 0.00235,0.00379,0.00508,0.00636,0.00773, 0.00902] #in A-1             
                    #width = 0.0002  
                    qth_interest = 1, #the intested single qth             
                    use_sqnorm = False,
                    use_imgsum_norm = True,

                    pdf_version = '_1' #for pdf report name    
                  )
           
    '''
    
    scat_geometry = run_pargs['scat_geometry']
    force_compress =  run_pargs['force_compress'] 
    para_compress = run_pargs['para_compress']
    run_fit_form = run_pargs['run_fit_form']
    run_waterfall = run_pargs['run_waterfall'] 
    run_t_ROI_Inten = run_pargs['run_t_ROI_Inten'] 
    #run_fit_g2 = run_pargs['run_fit_g2'], 
    fit_g2_func = run_pargs['fit_g2_func']
    run_one_time = run_pargs['run_one_time'] 
    run_two_time = run_pargs['run_two_time'] 
    run_four_time = run_pargs['run_four_time']
    run_xsvs=run_pargs['run_xsvs']
    ###############################################################
    if scat_geometry =='gi_saxs': #to be done for other types
        run_xsvs = False;
    ############################################################### 
    
    ###############################################################
    if scat_geometry == 'ang_saxs':
        run_xsvs= False;run_waterfall=False;run_two_time=False;run_four_time=False;run_t_ROI_Inten=False;    
    ###############################################################     
    if 'bin_frame' in list( run_pargs.keys() ):
        bin_frame = run_pargs['bin_frame']     
        bin_frame_number= run_pargs['bin_frame_number'] 
    else:
        bin_frame = False     
    if not bin_frame:
        bin_frame_number = 1 
    
    att_pdf_report = run_pargs['att_pdf_report'] 
    show_plot = run_pargs['show_plot']    
    CYCLE = run_pargs['CYCLE'] 
    mask_path = run_pargs['mask_path']
    mask_name = run_pargs['mask_name']
    good_start =     run_pargs['good_start'] 
    use_imgsum_norm =  run_pargs['use_imgsum_norm']
    try:
        use_sqnorm =     run_pargs['use_sqnorm']
    except:
        use_sqnorm = False
    try:
        inc_x0 = run_pargs['inc_x0']
        inc_y0 = run_pargs['inc_y0']
    except:
        inc_x0 = None
        inc_y0=  None 
        
    #for different scattering geogmetry, we only need to change roi_mask
    #and qval_dict
    qval_dict = run_pargs['qval_dict'] 
    if scat_geometry != 'ang_saxs':
        roi_mask = run_pargs['roi_mask']
        qind, pixelist = roi.extract_label_indices(  roi_mask  )
        noqs = len(np.unique(qind))
        nopr = np.bincount(qind, minlength=(noqs+1))[1:]
        
    else:        
        roi_mask_p = run_pargs['roi_mask_p']   
        qval_dict_p = run_pargs['qval_dict_p']
        roi_mask_v = run_pargs['roi_mask_v']   
        qval_dict_v = run_pargs['qval_dict_v']
        
    if scat_geometry == 'gi_saxs': 
        refl_x0 = run_pargs['refl_x0']
        refl_y0 = run_pargs['refl_y0']
        Qr, Qz, qr_map, qz_map  =   run_pargs['Qr'], run_pargs['Qz'], run_pargs['qr_map'], run_pargs['qz_map']
        
    
    taus=None;g2=None;tausb=None;g2b=None;g12b=None;taus4=None;g4=None;times_xsv=None;contrast_factorL=None;     
    qth_interest = run_pargs['qth_interest']    
    pdf_version = run_pargs['pdf_version']    
    
    
    username = getpass.getuser()
    data_dir0 = os.path.join('/XF11ID/analysis/', CYCLE, username, 'Results/')
    os.makedirs(data_dir0, exist_ok=True)
    print('Results from this analysis will be stashed in the directory %s' % data_dir0)
    #uid = (sys.argv)[1]
    print ('*'*40)
    print  ( '*'*5 + 'The processing uid is: %s'%uid + '*'*5)
    print ('*'*40)
    data_dir = os.path.join(data_dir0, '%s/'%uid)
    os.makedirs(data_dir, exist_ok=True)
    print('Results from this analysis will be stashed in the directory %s' % data_dir)
    md = get_meta_data( uid )
    uidstr = 'uid=%s'%uid
    imgs = load_data( uid, md['detector'], reverse= True  )
    md.update( imgs.md );Nimg = len(imgs);
    
    if inc_x0 is not None:
        md['beam_center_x']= inc_x0
    if inc_y0 is not None:
        md['beam_center_y']= inc_y0         
    center = [  int(md['beam_center_y']),int( md['beam_center_x'] ) ]  #beam center [y,x] for python image
    pixel_mask =  1- np.int_( np.array( imgs.md['pixel_mask'], dtype= bool)  )
    print( 'The data are: %s' %imgs )
    
    if False:
        print_dict( md,  ['suid', 'number of images', 'uid', 'scan_id', 'start_time', 'stop_time', 'sample', 'Measurement',
                  'acquire period', 'exposure time', 
         'det_distanc', 'beam_center_x', 'beam_center_y', ] )
        
    ## Overwrite Some Metadata if Wrong Input    
    dpix, lambda_, Ldet,  exposuretime, timeperframe, center = check_lost_metadata(
                md, Nimg, inc_x0 = inc_x0, inc_y0=   inc_y0, pixelsize = 7.5*10*(-5) )
    
    timeperframe  *=  bin_frame_number
    
    setup_pargs=dict(uid=uidstr, dpix= dpix, Ldet=Ldet, lambda_= lambda_, exposuretime=exposuretime,
        timeperframe=timeperframe, center=center, path= data_dir)
    #print_dict( setup_pargs )
    
    mask = load_mask(mask_path, mask_name, plot_ =  False, image_name = uidstr + '_mask', reverse=True ) 
    mask *= pixel_mask
    if md['detector'] =='eiger4m_single_image':
        mask[:,2069] =0 # False  #Concluded from the previous results
    show_img(mask,image_name = uidstr + '_mask', save=True, path=data_dir)
    mask_load=mask.copy()
    imgsa = apply_mask( imgs, mask )


    img_choice_N = 3
    img_samp_index = random.sample( range(len(imgs)), img_choice_N) 
    avg_img =  get_avg_img( imgsa, img_samp_index, plot_ = False, uid =uidstr)
    
    if avg_img.max() == 0:
        print('There are no photons recorded for this uid: %s'%uid)
        print('The data analysis should be terminated! Please try another uid.')
        
    else:   
        if scat_geometry !='saxs':
            show_img( avg_img,  vmin=.1, vmax=np.max(avg_img*.1), logs=True,
         image_name= uidstr + '_%s_frames_avg'%img_choice_N,  save=True, path=data_dir) 
        else:    
            show_saxs_qmap( avg_img, setup_pargs, width=400, show_pixel = False,
               vmin=.1, vmax= np.max(avg_img), logs=True, image_name= uidstr + '_%s_frames_avg'%img_choice_N )

        compress=True
        photon_occ = len( np.where(avg_img)[0] ) / ( imgsa[0].size)
        #compress =  photon_occ < .4  #if the photon ocupation < 0.5, do compress
        print ("The non-zeros photon occupation is %s."%( photon_occ))
        print("Will " + 'Always ' + ['NOT', 'DO'][compress]  + " apply compress process.")
        #good_start = 5  #make the good_start at least 0
        t0= time.time()
        filename = '/XF11ID/analysis/Compressed_Data' +'/uid_%s.cmp'%md['uid']
        mask, avg_img, imgsum, bad_frame_list = compress_eigerdata(imgs, mask, md, filename, 
                 force_compress= force_compress,  para_compress= para_compress,  bad_pixel_threshold= 1e14,
                                bins=bin_frame_number, num_sub= 100, num_max_para_process= 500  )
        min_inten = 10    
        good_start = max(good_start, np.where( np.array(imgsum) > min_inten )[0][0] )    
        print ('The good_start frame number is: %s '%good_start)
        FD = Multifile(filename, good_start, len(imgs))
        #FD = Multifile(filename, good_start, 100)
        uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
        print( uid_ )
        plot1D( y = imgsum[ np.array( [i for i in np.arange(good_start, len(imgsum)) if i not in bad_frame_list])],
               title =uidstr + '_imgsum', xlabel='Frame', ylabel='Total_Intensity', legend='imgsum'   )
        run_time(t0)

        #%system   free && sync && echo 3 > /proc/sys/vm/drop_caches && free
        ## Get bad frame list by a polynominal fit
        bad_frame_list =  get_bad_frame_list( imgsum, fit=True, plot=True,polyfit_order = 30, 
                        scale= 5.5,  good_start = good_start, uid= uidstr, path=data_dir)
        print( 'The bad frame list length is: %s'%len(bad_frame_list) )
        
        #### For beamline to find the bad pixels
        #bp = find_bad_pixels( FD, bad_frame_list, md['uid'] )
        #bp
        #bp.to_csv('/XF11ID/analysis/Commissioning/eiger4M_badpixel.csv', mode='a'  )
        
        ### Creat new mask by masking the bad pixels and get new avg_img
        if False:
            mask = mask_exclude_badpixel( bp, mask, md['uid'])
            avg_img = get_avg_imgc( FD,  sampling = 1, bad_frame_list=bad_frame_list )

        imgsum_y = imgsum[ np.array( [i for i in np.arange( len(imgsum)) if i not in bad_frame_list])]
        imgsum_x = np.arange( len( imgsum_y))
        save_lists(  [imgsum_x, imgsum_y], label=['Frame', 'Total_Intensity'],
           filename=uidstr + '_img_sum_t', path= data_dir  )
        plot1D( y = imgsum_y, title = uidstr + '_img_sum_t', xlabel='Frame',
               ylabel='Total_Intensity', legend='imgsum', save=True, path=data_dir)
        
        
        ############for SAXS and ANG_SAXS (Flow_SAXS)
        if scat_geometry =='saxs' or scat_geometry =='ang_saxs':
        
            show_saxs_qmap( avg_img, setup_pargs, width=600, vmin=.1, vmax=np.max(avg_img*.1), logs=True,
               image_name= uidstr + '_img_avg',  save=True)  
            #np.save(  data_dir + 'uid=%s--img-avg'%uid, avg_img)

            hmask = create_hot_pixel_mask( avg_img, threshold = 100, center=center, center_radius= 60)
            qp_saxs, iq_saxs, q_saxs = get_circular_average( avg_img, mask * hmask, pargs=setup_pargs  )
        
            plot_circular_average( qp_saxs, iq_saxs, q_saxs,  pargs= setup_pargs, 
                              xlim=[q_saxs.min(), q_saxs.max()], ylim = [iq_saxs.min(), iq_saxs.max()] )

            pd = trans_data_to_pd( np.where( hmask !=1), 
                    label=[md['uid']+'_hmask'+'x', md['uid']+'_hmask'+'y' ], dtype='list')
            
            #pd.to_csv('/XF11ID/analysis/Commissioning/eiger4M_badpixel.csv', mode='a'  )
            
            mask =np.array( mask * hmask, dtype=bool) 
            #show_img( mask )
         
            if run_fit_form:
                form_res = fit_form_factor( q_saxs,iq_saxs,  guess_values={'radius': 2500, 'sigma':0.05, 
                 'delta_rho':1E-10 },  fit_range=[0.0001, 0.015], fit_variables={'radius': T, 'sigma':T, 
                 'delta_rho':T},  res_pargs=setup_pargs, xlim=[0.0001, 0.015])

            show_ROI_on_image( avg_img, roi_mask, center, label_on = False, rwidth =700, alpha=.9,  
                             save=True, path=data_dir, uid=uidstr, vmin= np.min(avg_img), vmax= np.max(avg_img) ) 
            
            qr = np.array( [ qval_dict[k][0] for k in list( qval_dict.keys()) ] )
            plot_qIq_with_ROI( q_saxs, iq_saxs, qr, logs=True, uid=uidstr, xlim=[q_saxs.min(), q_saxs.max()],
                      ylim = [iq_saxs.min(), iq_saxs.max()],  save=True, path=data_dir)
            
            if  scat_geometry != 'ang_saxs':     
                Nimg = FD.end - FD.beg 
                time_edge = create_time_slice( N= Nimg, slice_num= 3, slice_width= 1, edges = None )
                time_edge =  np.array( time_edge ) + good_start
                #print( time_edge )
                qpt, iqst, qt = get_t_iqc( FD, time_edge, mask, pargs=setup_pargs, nx=1500 )
                plot_t_iqc( qt, iqst, time_edge, pargs=setup_pargs, xlim=[qt.min(), qt.max()],
                       ylim = [iqst.min(), iqst.max()], save=True )   
                
        elif scat_geometry == 'gi_waxs':  
            #roi_mask[badpixel] = 0   
            qr = np.array( [ qval_dict[k][0] for k in list( qval_dict.keys()) ] )
            show_ROI_on_image( avg_img, roi_mask, label_on = True,  alpha=.5,save=True, path= data_dir, uid=uidstr)#, vmin=1, vmax=15)

        elif scat_geometry == 'gi_saxs':            
            show_img( avg_img,  vmin=.1, vmax=np.max(avg_img*.1),
                     logs=True, image_name= uidstr + '_img_avg',  save=True, path=data_dir)            
            ticks_  = get_qzr_map(  qr_map, qz_map, inc_x0, Nzline=10,  Nrline=10   )
            ticks = ticks_[:4]            
            plot_qzr_map(  qr_map, qz_map, inc_x0, ticks = ticks_, data= avg_img, uid= uidstr, path = data_dir   ) 
            show_qzr_roi( avg_img, roi_mask, inc_x0, ticks, alpha=0.5, save=True, path=data_dir, uid=uidstr )            
            qr_1d_pds = cal_1d_qr( avg_img, Qr, Qz, qr_map, qz_map, inc_x0,  setup_pargs=setup_pargs )            
            plot_qr_1d_with_ROI( qr_1d_pds, qr_center=np.unique( np.array(list( qval_dict.values() ) )[:,0] ),
                    loglog=False, save=True, uid=uidstr, path = data_dir)
            
            Nimg = FD.end - FD.beg 
            time_edge = create_time_slice( N= Nimg, slice_num= 3, slice_width= 1, edges = None )
            time_edge =  np.array( time_edge ) + good_start
            qrt_pds = get_t_qrc( FD, time_edge, Qr, Qz, qr_map, qz_map, path=data_dir, uid = uidstr )    
            plot_qrt_pds( qrt_pds, time_edge, qz_index = 0, uid = uidstr, path =  data_dir )
            
            

        ##############################
        ##the below works for all the geometries     
        ########################################
        if scat_geometry !='ang_saxs':    
            roi_inten = check_ROI_intensity( avg_img, roi_mask, ring_number= qth_interest, uid =uidstr, save=True, path=data_dir ) 
        if scat_geometry =='saxs' or scat_geometry =='gi_saxs' or scat_geometry =='gi_waxs':
            if run_waterfall:
                qindex = qth_interest
                wat = cal_waterfallc( FD, roi_mask, qindex= qindex, save =True, path=data_dir,uid=uidstr)
            if run_waterfall: 
                plot_waterfallc( wat, qindex, aspect=None, 
                                vmax=  np.max(wat), uid=uidstr, save =True, 
                                path=data_dir, beg= FD.beg)
        ring_avg = None    
        if run_t_ROI_Inten:
            times_roi, mean_int_sets = cal_each_ring_mean_intensityc(FD, roi_mask, timeperframe = None,  ) 
            plot_each_ring_mean_intensityc( times_roi, mean_int_sets,  uid = uidstr, save=True, path=data_dir )
            roi_avg = np.average( mean_int_sets, axis=0)

        uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
        lag_steps = None
        
        if use_sqnorm:
            norm = get_pixelist_interp_iq( qp_saxs, iq_saxs, roi_mask, center)
        else:
            norm=None 
            
        define_good_series = False
        if define_good_series:
            FD = Multifile(filename, beg = good_start, end = Nimg)
            uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
            print( uid_ )
    
        if run_one_time: 
            if use_imgsum_norm:
                imgsum_ = imgsum
            else:
                imgsum_ = None 
            if scat_geometry !='ang_saxs':  
                t0 = time.time()
                g2, lag_steps  = cal_g2p( FD,  roi_mask, bad_frame_list,good_start, num_buf = 8, num_lev= None,
                                    imgsum= imgsum_, norm=norm )
                run_time(t0)
                taus = lag_steps * timeperframe    
                g2_pds = save_g2_general( g2, taus=taus,qr=np.array( list( qval_dict.values() ) )[:,0],
                                     uid=uid_+'_g2.csv', path= data_dir, return_res=True )
                
                g2_fit_result, taus_fit, g2_fit = get_g2_fit_general( g2,  taus, 
                        function = fit_g2_func,  vlim=[0.95, 1.05], fit_range= None,  
                    fit_variables={'baseline':True, 'beta':True, 'alpha':False,'relaxation_rate':True},                                  
                    guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,}) 
                g2_fit_paras = save_g2_fit_para_tocsv(g2_fit_result,  filename= uid_  +'_g2_fit_paras.csv', path=data_dir ) 
                
                #if run_one_time:
                #plot_g2_general( g2_dict={1:g2}, taus_dict={1:taus},vlim=[0.95, 1.05], qval_dict = qval_dict, fit_res= None, 
                #                geometry='saxs',filename=uid_+'--g2',path= data_dir,   ylabel='g2')            
                
                plot_g2_general( g2_dict={1:g2, 2:g2_fit}, taus_dict={1:taus, 2:taus_fit},vlim=[0.95, 1.05],
                qval_dict = qval_dict, fit_res= g2_fit_result,  geometry=scat_geometry,filename=uid_ + '_g2', 
                    path= data_dir, function= fit_g2_func,  ylabel='g2', append_name=  '_fit')

                D0, qrate_fit_res = get_q_rate_fit_general(  qval_dict, g2_fit_paras['relaxation_rate'], geometry= scat_geometry )
                plot_q_rate_fit_general( qval_dict, g2_fit_paras['relaxation_rate'],  qrate_fit_res, 
                            geometry= scat_geometry,uid=uid_  , path= data_dir )            
                               
                
            else:
                t0 = time.time()                
                g2_v, lag_steps_v  = cal_g2p( FD,  roi_mask_v, bad_frame_list,good_start, num_buf = 8, num_lev= None,
                                    imgsum= imgsum_, norm=norm ) 
                g2_p, lag_steps_p  = cal_g2p( FD,  roi_mask_p, bad_frame_list,good_start, num_buf = 8, num_lev= None,
                                    imgsum= imgsum_, norm=norm )                 
                run_time(t0)            

                taus_v = lag_steps_v * timeperframe    
                g2_pds_v = save_g2_general( g2_v, taus=taus_v,qr=np.array( list( qval_dict_v.values() ) )[:,0],
                                     uid=uid_+'_g2v.csv', path= data_dir, return_res=True )       

                taus_p = lag_steps_p * timeperframe    
                g2_pds_p = save_g2_general( g2_p, taus=taus_p,qr=np.array( list( qval_dict_p.values() ) )[:,0],
                                     uid=uid_+'_g2p.csv', path= data_dir, return_res=True )  
                
                fit_g2_func_v = 'stretched' #for vertical
                g2_fit_result_v, taus_fit_v, g2_fit_v = get_g2_fit_general( g2_v,  taus_v, 
                        function = fit_g2_func_v,  vlim=[0.95, 1.05], fit_range= None,  
                    fit_variables={'baseline':True, 'beta':True, 'alpha':False,'relaxation_rate':True},                                  
                    guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,}) 
                g2_fit_paras_v = save_g2_fit_para_tocsv(g2_fit_result_v,  filename= uid_  +'_g2_fit_paras_v.csv', path=data_dir ) 

                fit_g2_func_p ='flow_para' #for parallel
                g2_fit_result_p, taus_fit_p, g2_fit_p = get_g2_fit_general( g2_p,  taus_p, 
                        function = fit_g2_func_p,  vlim=[0.95, 1.05], fit_range= None,  
                    fit_variables={'baseline':True, 'beta':True, 'alpha':False,'relaxation_rate':True,'flow_velocity':True},                                  
                    guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,'flow_velocity':1}) 
                g2_fit_paras_p = save_g2_fit_para_tocsv(g2_fit_result_p,  filename= uid_  +'_g2_fit_paras_p.csv', path=data_dir )             
            
           

                plot_g2_general( g2_dict={1:g2_v, 2:g2_fit_v}, taus_dict={1:taus_v, 2:taus_fit_v},vlim=[0.95, 1.05],
                        qval_dict = qval_dict_v, fit_res= g2_fit_result_v,  geometry=scat_geometry,filename= uid_+'_g2_v', 
                            path= data_dir, function= fit_g2_func_v,  ylabel='g2_v', append_name=  '_fit')        

                plot_g2_general( g2_dict={1:g2_p, 2:g2_fit_p}, taus_dict={1:taus_p, 2:taus_fit_p},vlim=[0.95, 1.05],
                        qval_dict = qval_dict_p, fit_res= g2_fit_result_p,  geometry=scat_geometry,filename= uid_+'_g2_p', 
                            path= data_dir, function= fit_g2_func_p,  ylabel='g2_p', append_name=  '_fit')   

                combine_images( [data_dir + uid_+'_g2_v_fit.png', data_dir + uid_+'_g2_p_fit.png'], data_dir + uid_+'_g2_fit.png', outsize=(2000, 2400) )
                
                
                D0_v, qrate_fit_res_v = get_q_rate_fit_general(  qval_dict_v, g2_fit_paras_v['relaxation_rate'], geometry= scat_geometry )
                plot_q_rate_fit_general( qval_dict_v, g2_fit_paras_v['relaxation_rate'],  qrate_fit_res_v, 
                            geometry= scat_geometry,uid=uid_ +'_vert' , path= data_dir )
        
                D0_p, qrate_fit_res_p = get_q_rate_fit_general(  qval_dict_p, g2_fit_paras_p['relaxation_rate'], geometry= scat_geometry )
                plot_q_rate_fit_general( qval_dict_p, g2_fit_paras_p['relaxation_rate'],  qrate_fit_res_p, 
                            geometry= scat_geometry,uid=uid_ +'_para' , path= data_dir )
        
        
                combine_images( [data_dir + uid_+ '_vert_Q_Rate_fit.png', data_dir + uid_+ '_para_Q_Rate_fit.png'], data_dir + uid_+'_Q_Rate_fit.png', outsize=(2000, 2400) )


        # For two-time
        data_pixel = None
        if run_two_time:    
            data_pixel =   Get_Pixel_Arrayc( FD, pixelist,  norm=norm ).get_data()
            t0=time.time()    
            g12b = auto_two_Arrayc(  data_pixel,  roi_mask, index = None   )
            if lag_steps is None:
                num_bufs=8
                noframes = FD.end - FD.beg
                num_levels = int(np.log( noframes/(num_bufs-1))/np.log(2) +1) +1
                tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)
                max_taus= lag_steps.max()
            run_time( t0 )    

            show_C12(g12b, q_ind= qth_interest, N1= FD.beg, N2=min( FD.end,1000), vmin=1.1, vmax=1.25,
                     timeperframe=timeperframe,save=True,
                     path= data_dir, uid = uid_ ) 
            
            max_taus = Nimg    
            t0=time.time()            
            #g2b = get_one_time_from_two_time(g12b)[:max_taus]
            g2b = get_one_time_from_two_time(g12b)[lag_steps]
            tausb = lag_steps *timeperframe
            run_time(t0)            
            #tausb = np.arange( g2b.shape[0])[:max_taus] *timeperframe            
            g2b_pds = save_g2_general( g2b, taus=tausb, qr= np.array( list( qval_dict.values() ) )[:,0],
                                      qz=None, uid=uid_ +'_g2b.csv', path= data_dir, return_res=True ) 
            
            g2_fit_resultb, taus_fitb, g2_fitb = get_g2_fit_general( g2b,  tausb, 
                function = fit_g2_func,  vlim=[0.95, 1.05], fit_range= None,  
                fit_variables={'baseline':True, 'beta':True, 'alpha':False,'relaxation_rate':True},
                guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,}) 
    
            g2b_fit_paras = save_g2_fit_para_tocsv(g2_fit_resultb, 
                    filename= uid_  + '_g2b_fit_paras.csv', path=data_dir )
    
            
            plot_g2_general( g2_dict={1:g2b, 2:g2_fitb}, taus_dict={1:tausb, 2:taus_fitb},vlim=[0.95, 1.05],
                qval_dict=qval_dict, fit_res= g2_fit_resultb,  geometry=scat_geometry,filename=uid_+'_g2', 
                    path= data_dir, function= fit_g2_func,  ylabel='g2', append_name=  '_b_fit')
            
            if run_two_time and run_one_time:
                plot_g2_general( g2_dict={1:g2, 2:g2b}, taus_dict={1:taus, 2:tausb},vlim=[0.95, 1.05],
                qval_dict=qval_dict, g2_labels=['from_one_time', 'from_two_time'],
            geometry=scat_geometry,filename=uid_+'_g2_two_g2', path= data_dir, ylabel='g2', )

        # Four Time Correlation

        if run_four_time: #have to run one and two first
            t0=time.time()
            g4 = get_four_time_from_two_time(g12b, g2=g2b)[:max_taus]
            run_time(t0)

            taus4 = np.arange( g4.shape[0])*timeperframe        
            g4_pds = save_g2_general( g4, taus=taus4, qr=np.array( list( qval_dict.values() ) )[:,0],
                                     qz=None, uid=uid_ +'_g4.csv', path= data_dir, return_res=True )
            plot_g2_general( g2_dict={1:g4}, taus_dict={1:taus4},vlim=[0.95, 1.05], qval_dict=qval_dict, fit_res= None, 
                        geometry=scat_geometry,filename=uid_+'_g4',path= data_dir,   ylabel='g4')

        # Speckel Visiblity   
        if run_xsvs:    
            max_cts = get_max_countc(FD, roi_mask )
            qind, pixelist = roi.extract_label_indices(   roi_mask  )
            noqs = len( np.unique(qind) )
            nopr = np.bincount(qind, minlength=(noqs+1))[1:]
            #time_steps = np.array( utils.geometric_series(2,   len(imgs)   ) )
            time_steps = [0,1]  #only run the first two levels
            num_times = len(time_steps)    
            times_xsvs = exposuretime + (2**(  np.arange( len(time_steps) ) ) -1 ) *timeperframe    
            print( 'The max counts are: %s'%max_cts )

        ### Do historam 
            if roi_avg is  None:
                times_roi, mean_int_sets = cal_each_ring_mean_intensityc(FD, roi_mask, timeperframe = None,  ) 
                roi_avg = np.average( mean_int_sets, axis=0)

            t0=time.time()
            spec_bins, spec_his, spec_std  =  xsvsp( FD, np.int_(roi_mask), norm=None,
                        max_cts=int(max_cts+2),  bad_images=bad_frame_list, only_two_levels=True )    
            spec_kmean =  np.array(  [roi_avg * 2**j for j in  range( spec_his.shape[0] )] )
            run_time(t0)
            
            run_xsvs_all_lags = False
            if run_xsvs_all_lags:
                times_xsvs = exposuretime +  lag_steps * acquisition_period
                if data_pixel is None:
                    data_pixel =   Get_Pixel_Arrayc( FD, pixelist,  norm=norm ).get_data()       
                t0=time.time()
                spec_bins, spec_his, spec_std, spec_kmean = get_binned_his_std(data_pixel, np.int_(ro_mask), lag_steps )
                run_time(t0) 
            spec_pds =  save_bin_his_std( spec_bins, spec_his, spec_std, filename=uid_+'_spec_res.csv', path=data_dir ) 

            ML_val, KL_val,K_ = get_xsvs_fit(  spec_his, spec_kmean,  spec_std, max_bins=2,varyK= False, )

            #print( 'The observed average photon counts are: %s'%np.round(K_mean,4))
            #print( 'The fitted average photon counts are: %s'%np.round(K_,4)) 
            print( 'The difference sum of average photon counts between fit and data are: %s'%np.round( 
                    abs(np.sum( spec_kmean[0,:] - K_ )),4))
            print( '#'*30)
            qth=   10 
            print( 'The fitted M for Qth= %s are: %s'%(qth, ML_val[qth]) )
            print( K_[qth])
            print( '#'*30)


            plot_xsvs_fit(  spec_his, ML_val, KL_val, K_mean = spec_kmean, spec_std=spec_std,
                  xlim = [0,10], vlim =[.9, 1.1],
        uid=uid_, qth= qth_interest, logy= True, times= times_xsvs, q_ring_center=qr, path=data_dir)
            
            plot_xsvs_fit(  spec_his, ML_val, KL_val, K_mean = spec_kmean, spec_std = spec_std,
                  xlim = [0,15], vlim =[.9, 1.1],
        uid=uid_, qth= None, logy= True, times= times_xsvs, q_ring_center=qr, path=data_dir )                

            ### Get contrast
            contrast_factorL = get_contrast( ML_val)
            spec_km_pds = save_KM(  spec_kmean, KL_val, ML_val, qs=qr, level_time=times_xsvs, uid=uid_ , path = data_dir )
            #print( spec_km_pds )
    
            plot_g2_contrast( contrast_factorL, g2, times_xsvs, taus, qr, 
                             vlim=[0.8,1.2], qth = qth_interest, uid=uid_,path = data_dir, legend_size=14)

            plot_g2_contrast( contrast_factorL, g2, times_xsvs, taus, qr, 
                             vlim=[0.8,1.2], qth = None, uid=uid_,path = data_dir, legend_size=4)
   

        
    
        md['mask_file']= mask_path + mask_name
        md['mask'] = mask
        md['NOTEBOOK_FULL_PATH'] = None
        md['good_start'] = good_start
        md['bad_frame_list'] = bad_frame_list
        md['avg_img'] = avg_img
        md['roi_mask'] = roi_mask

        if scat_geometry == 'gi_saxs':        
            md['Qr'] = Qr
            md['Qz'] = Qz
            md['qval_dict'] = qval_dict
            md['beam_center_x'] =  inc_x0
            md['beam_center_y']=   inc_y0
            md['beam_refl_center_x'] = refl_x0
            md['beam_refl_center_y'] = refl_y0

        elif  scat_geometry == 'saxs' or 'gi_waxs':
            md['qr']= qr
            #md['qr_edge'] = qr_edge
            md['qval_dict'] = qval_dict
            md['beam_center_x'] =  center[1]
            md['beam_center_y']=  center[0]  

        elif  scat_geometry == 'ang_saxs':
            md['qval_dict_v'] = qval_dict_v
            md['qval_dict_p'] = qval_dict_p
            md['beam_center_x'] =  center[1]
            md['beam_center_y']=  center[0]  
            

        md['beg'] = FD.beg
        md['end'] = FD.end
        md['metadata_file'] = data_dir + 'md.csv-&-md.pkl'
        psave_obj(  md, data_dir + 'uid=%s_md'%uid ) #save the setup parameters
        save_dict_csv( md,  data_dir + 'uid=%s_md.csv'%uid, 'w')

        Exdt = {} 
        if scat_geometry == 'gi_saxs':  
            for k,v in zip( ['md', 'roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list', 'qr_1d_pds'], 
                        [md,    roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list, qr_1d_pds] ):
                Exdt[ k ] = v
        elif scat_geometry == 'saxs': 
            for k,v in zip( ['md', 'q_saxs', 'iq_saxs','iqst','qt','roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                        [md, q_saxs, iq_saxs, iqst, qt,roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
                Exdt[ k ] = v
        elif scat_geometry == 'gi_waxs': 
            for k,v in zip( ['md', 'roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                        [md, roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
                Exdt[ k ] = v                
        elif scat_geometry == 'ang_saxs':     
            for k,v in zip( ['md', 'q_saxs', 'iq_saxs','roi_mask_v','roi_mask_p',
                             'qval_dict_v','qval_dict_p','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                       [md, q_saxs, iq_saxs, roi_mask_v,roi_mask_p,
                        qval_dict_v,qval_dict_p, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
                Exdt[ k ] = v 

        if run_waterfall:Exdt['wat'] =  wat
        if run_t_ROI_Inten:Exdt['times_roi'] = times_roi;Exdt['mean_int_sets']=mean_int_sets
        if run_one_time:
            if scat_geometry != 'ang_saxs':         
                for k,v in zip( ['taus','g2','g2_fit_paras'], [taus,g2,g2_fit_paras] ):Exdt[ k ] = v
            else:
                for k,v in zip( ['taus_v','g2_v','g2_fit_paras_v'], [taus_v,g2_v,g2_fit_paras_v] ):Exdt[ k ] = v
                for k,v in zip( ['taus_p','g2_p','g2_fit_paras_p'], [taus_p,g2_p,g2_fit_paras_p] ):Exdt[ k ] = v
        if run_two_time:
            for k,v in zip( ['tausb','g2b','g2b_fit_paras', 'g12b'], [tausb,g2b,g2b_fit_paras,g12b] ):Exdt[ k ] = v
        if run_four_time:
            for k,v in zip( ['taus4','g4'], [taus4,g4] ):Exdt[ k ] = v
        if run_xsvs:
            for k,v in zip( ['spec_kmean','spec_pds','times_xsvs','spec_km_pds','contrast_factorL'], 
                           [ spec_kmean,spec_pds,times_xsvs,spec_km_pds,contrast_factorL] ):Exdt[ k ] = v 

                
        export_xpcs_results_to_h5( 'uid=%s_Res.h5'%md['uid'], data_dir, export_dict = Exdt )
        #extract_dict = extract_xpcs_results_from_h5( filename = 'uid=%s_Res.h5'%md['uid'], import_dir = data_dir )        
        # Creat PDF Report
        pdf_out_dir = os.path.join('/XF11ID/analysis/', CYCLE, username, 'Results/')     
        pdf_filename = "XPCS_Analysis_Report_for_uid=%s%s.pdf"%(uid,pdf_version)
        if run_xsvs:
            pdf_filename = "XPCS_XSVS_Analysis_Report_for_uid=%s%s.pdf"%(uid,pdf_version)
        #pdf_filename
        make_pdf_report( data_dir, uid, pdf_out_dir, pdf_filename, username, 
                        run_fit_form, run_one_time, run_two_time, run_four_time, run_xsvs, report_type= scat_geometry
                       ) 
        ## Attach the PDF report to Olog 
        if att_pdf_report:
            os.environ['HTTPS_PROXY'] = 'https://proxy:8888'
            os.environ['no_proxy'] = 'cs.nsls2.local,localhost,127.0.0.1'
            pname = pdf_out_dir + pdf_filename 
            atch=[  Attachment(open(pname, 'rb')) ] 
            try:
                update_olog_uid( uid= md['uid'], text='Add XPCS Analysis PDF Report', attachments= atch )
            except:
                print("I can't attach this PDF: %s due to a duplicated filename. Please give a different PDF file."%pname)

        if show_plot:
            plt.show()   
        #else:
        #    plt.close('all')

        if return_res:
            res = {}
            if scat_geometry == 'saxs': 
                for k,v in zip( ['md', 'q_saxs', 'iq_saxs','iqst','qt','avg_img','mask', 'imgsum','bad_frame_list','roi_mask', 'qval_dict'], 
                        [ md, q_saxs, iq_saxs, iqst, qt, avg_img,mask,imgsum,bad_frame_list,roi_mask, qval_dict ] ):
                    res[ k ] = v
                    
            elif scat_geometry == 'ang_saxs': 
                for k,v in zip( [ 'md', 'q_saxs', 'iq_saxs','roi_mask_v','roi_mask_p',
                             'qval_dict_v','qval_dict_p','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                       [ md, q_saxs, iq_saxs, roi_mask_v,roi_mask_p,
                        qval_dict_v,qval_dict_p, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
                    res[ k ] = v                 
                
            elif scat_geometry == 'gi_saxs': 
                for k,v in zip( ['md', 'roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list', 'qr_1d_pds'], 
                        [md,    roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list, qr_1d_pds] ):
                    res[ k ] = v
                    
            elif scat_geometry == 'gi_waxs': 
                for k,v in zip( ['md', 'roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                        [md, roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
                    res[ k ] = v                      
                
            if run_waterfall:
                res['wat'] =  wat
            if run_t_ROI_Inten:
                res['times_roi'] = times_roi;
                res['mean_int_sets']=mean_int_sets            
            if run_one_time:
                if scat_geometry != 'ang_saxs': 
                    res['g2'] = g2
                    res['taus']=taus
                else:
                    res['g2_p'] = g2_p
                    res['taus_p']=taus_p
                    res['g2_v'] = g2_v
                    res['taus_v']=taus_v
                    
            if run_two_time:
                res['tausb'] = tausb
                res['g12b'] = g12b
                res['g2b'] = g2b
            if run_four_time:
                res['g4']= g4
                res['taus4']=taus4
            if run_xsvs:
                res['spec_kmean']=spec_kmean
                res['spec_pds']= spec_pds
                res['contrast_factorL'] = contrast_factorL 
                res['times_xsvs']= times_xsvs
            return res
    
#uid = '3ff4ee'
#run_xpcs_xsvs_single( uid,  run_pargs )





