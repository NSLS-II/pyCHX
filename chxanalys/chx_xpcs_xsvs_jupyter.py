from chxanalys.chx_packages import *

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
    att_pdf_report = run_pargs['att_pdf_report'] 
    show_plot = run_pargs['show_plot']    
    CYCLE = run_pargs['CYCLE'] 
    mask_path = run_pargs['mask_path']
    mask_name = run_pargs['mask_name']
    good_start =     run_pargs['good_start']     
    uniformq =      run_pargs['uniformq']
    use_imgsum_norm =  run_pargs['use_imgsum_norm']
    try:        
        inner_radius= run_pargs['inner_radius']
        outer_radius =   run_pargs['outer_radius']
        gap_ring_number =run_pargs['gap_ring_number']
        num_rings =   run_pargs['num_rings']  
    except:
        number_rings= run_pargs['number_rings']  
        width = run_pargs['width']
        qcenters = run_pargs['qcenters']  
        
    taus=None;g2=None;tausb=None;g2b=None;g12b=None;taus4=None;g4=None;times_xsv=None;contrast_factorL=None; 
    
    qth_interest = run_pargs['qth_interest']
    use_sqnorm = run_pargs['use_sqnorm']
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

    imgs = load_data( uid, md['detector'], reverse= True  )
    md.update( imgs.md );Nimg = len(imgs);
    pixel_mask =  1- np.int_( np.array( imgs.md['pixel_mask'], dtype= bool)  )
    print( 'The data are: %s' %imgs )
    if False:
        print_dict( md,  ['suid', 'number of images', 'uid', 'scan_id', 'start_time', 'stop_time', 'sample', 'Measurement',
                  'acquire period', 'exposure time', 
         'det_distanc', 'beam_center_x', 'beam_center_y', ] )

    ## Overwrite Some Metadata if Wrong Input
    dpix = md['x_pixel_size'] * 1000.  #in mm, eiger 4m is 0.075 mm
    lambda_ =md['incident_wavelength']    # wavelegth of the X-rays in Angstroms
    Ldet = md['detector_distance'] *1000     # detector to sample distance (mm)
    try:
        exposuretime= md['cam_acquire_t']     #exposure time in sec
    except:    
        exposuretime= md['count_time']     #exposure time in sec
    acquisition_period = md['frame_time']   #acquisition time in sec
    timeperframe = acquisition_period #for g2
    center = [  int(md['beam_center_y']),int( md['beam_center_x'] ) ]  #beam center [y,x] for python image
    setup_pargs=dict(uid=uid, dpix= dpix, Ldet=Ldet, lambda_= lambda_, exposuretime=exposuretime,
            timeperframe=timeperframe, center=center, path= data_dir)
    print_dict( setup_pargs )

    mask = load_mask(mask_path, mask_name, plot_ =  False, image_name = 'uid=%s-mask'%uid, reverse=True ) 
    mask *= pixel_mask
    mask[:,2069] =0 # False  #Concluded from the previous results
    #np.save(  data_dir + 'mask', mask)
    show_img(mask,image_name = 'uid=%s-mask'%uid, save=True, path=data_dir)
    mask_load=mask.copy()
    imgsa = apply_mask( imgs, mask )

    img_choice_N = 10
    img_samp_index = random.sample( range(len(imgs)), img_choice_N) 
    avg_img =  get_avg_img( imgsa, img_samp_index, plot_ = False, uid =uid)
    
    if avg_img.max() == 0:
        print('There are no photons recorded for this uid: %s'%uid)
        print('The data analysis should be terminated! Please try another uid.')
        
    else:        
        show_saxs_qmap( avg_img, setup_pargs, width=600, show_pixel = True,
                   vmin=.1, vmax= np.max(avg_img), logs=True, image_name= uid + '--%s-frames_avg'%img_choice_N )


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
                                bins=1, num_sub= 100, num_max_para_process= 500  )
        min_inten = 10    
        good_start = max(good_start, np.where( np.array(imgsum) > min_inten )[0][0] )    
        print ('The good_start frame number is: %s '%good_start)
        FD = Multifile(filename, good_start, len(imgs))
        plot1D( y = imgsum[ np.array( [i for i in np.arange(good_start, len(imgsum)) if i not in bad_frame_list])],
               title ='Uid= %s--imgsum'%uid, xlabel='Frame', ylabel='Total_Intensity', legend='imgsum'   )
        run_time(t0)
        #%system   free && sync && echo 3 > /proc/sys/vm/drop_caches && free
        ## Get bad frame list by a polynominal fit
        bad_frame_list =  get_bad_frame_list( imgsum, fit=True, plot=True,polyfit_order = 30, 
                            scale= 5.5,  good_start = good_start, uid=uid, path=data_dir)
        print( 'The bad frame list length is: %s'%len(bad_frame_list) )
        #### For beamline to find the bad pixels
        bp = find_bad_pixels( FD, bad_frame_list, md['uid'] )
        #bp
        bp.to_csv('/XF11ID/analysis/Commissioning/eiger4M_badpixel.csv', mode='a'  )
        ### Creat new mask by masking the bad pixels and get new avg_img
        if False:
            mask = mask_exclude_badpixel( bp, mask, md['uid'])
            avg_img = get_avg_imgc( FD,  sampling = 1, bad_frame_list=bad_frame_list )

        imgsum_y = imgsum[ np.array( [i for i in np.arange( len(imgsum)) if i not in bad_frame_list])]
        imgsum_x = np.arange( len( imgsum_y))
        save_lists(  [imgsum_x, imgsum_y], label=['Frame', 'Total_Intensity'],
                   filename='uid=%s--img-sum-t'%uid, path= data_dir  )
        plot1D( y = imgsum_y, title ='uid=%s--img-sum-t'%uid, xlabel='Frame',
               ylabel='Total_Intensity', legend='imgsum', save=True, path=data_dir)
        show_saxs_qmap( avg_img, setup_pargs, width=600,vmin=.1, vmax=np.max(avg_img*.1), logs=True,
                       image_name= 'uid=%s--img-avg'%uid,  save=True) 
        #np.save(  data_dir + 'uid=%s--img-avg'%uid, avg_img)

        hmask = create_hot_pixel_mask( avg_img, threshold = 100, center=center, center_radius= 400)
        qp_saxs, iq_saxs, q_saxs = get_circular_average( avg_img, mask * hmask, pargs=setup_pargs  )
        plot_circular_average( qp_saxs, iq_saxs, q_saxs,  pargs=setup_pargs, 
                      xlim=[q_saxs.min(), q_saxs.max()], ylim = [iq_saxs.min(), iq_saxs.max()] )

        pd = trans_data_to_pd( np.where( hmask !=1), 
                    label=[md['uid']+'_hmask'+'x', md['uid']+'_hmask'+'y' ], dtype='list')
        pd.to_csv('/XF11ID/analysis/Commissioning/eiger4M_badpixel.csv', mode='a'  )
        mask =np.array( mask * hmask, dtype=bool) 
        if run_fit_form:
            form_res = fit_form_factor( q_saxs,iq_saxs,  guess_values={'radius': 2500, 'sigma':0.05, 
                 'delta_rho':1E-10 },  fit_range=[0.0001, 0.015], fit_variables={'radius': T, 'sigma':T, 
                 'delta_rho':T},  res_pargs=setup_pargs, xlim=[0.0001, 0.015]) 
        ### Define a non-uniform distributed rings by giving edges
        if not uniformq:    
            #width = 0.0002    
            #number_rings= 1    
            #qcenters = [ 0.00235,0.00379,0.00508,0.00636,0.00773, 0.00902] #in A-1
            edges = get_non_uniform_edges(  qcenters, width, number_rings )    
            inner_radius= None
            outer_radius = None
            width = None
            num_rings = None
        if uniformq:    
            #inner_radius= 0.006  #16
            #outer_radius = 0.05  #112    
            #num_rings = 12
            width =    ( outer_radius - inner_radius)/(num_rings + gap_ring_number)
            edges = None 

        ring_mask, q_ring_center, q_ring_val = get_ring_mask(  mask, inner_radius=inner_radius, outer_radius = outer_radius , width = width, num_rings = num_rings, edges=edges,unit='A', pargs=setup_pargs   )

        qind, pixelist = roi.extract_label_indices(  ring_mask  ) 
        q_ring_center = np.round( q_ring_center, 4)
        show_ROI_on_image( avg_img, ring_mask, center, label_on = False, rwidth =700, alpha=.9,  
                     save=True, path=data_dir, uid=uid, vmin= np.min(avg_img), vmax= np.max(avg_img) )    
        plot_qIq_with_ROI( q_saxs, iq_saxs, q_ring_center, logs=True, uid=uid, xlim=[q_saxs.min(), q_saxs.max()],
                  ylim = [iq_saxs.min(), iq_saxs.max()],  save=True, path=data_dir)
        roi_inten = check_ROI_intensity( avg_img, ring_mask, ring_number= qth_interest, uid =uid, save=True, path=data_dir )


        Nimg = FD.end - FD.beg 
        time_edge = create_time_slice( N= Nimg, slice_num= 3, slice_width= 1, edges = None )
        time_edge =  np.array( time_edge ) + good_start
        print( time_edge )
        qpt, iqst, qt = get_t_iqc( FD, time_edge, mask, pargs=setup_pargs, nx=1500 )
        plot_t_iqc( qt, iqst, time_edge, pargs=setup_pargs, xlim=[qt.min(), qt.max()],
                   ylim = [iqst.min(), iqst.max()], save=True )
        if run_waterfall:
            qindex = qth_interest
            wat = cal_waterfallc( FD, ring_mask, qindex= qindex, save =True, path=data_dir, uid=uid)
        if run_waterfall: 
            plot_waterfallc( wat, qindex, aspect=None, 
                                vmax=  np.max(wat), uid=uid, save =True, 
                                path=data_dir, beg= FD.beg)
        ring_avg = None    
        if run_t_ROI_Inten:
            times_roi, mean_int_sets = cal_each_ring_mean_intensityc(FD, ring_mask, timeperframe = None,  ) 
            plot_each_ring_mean_intensityc( times_roi, mean_int_sets,  uid = uid, save=True, path=data_dir )
            ring_avg = np.average( mean_int_sets, axis=0)

        uid_ = uid + '--fra-%s-%s'%(FD.beg, FD.end) 
        lag_steps = None
        if use_sqnorm:
            norm = get_pixelist_interp_iq( qp_saxs, iq_saxs, ring_mask, center)
        else:
            norm=None 
            
        define_good_series = False
        if define_good_series:
            FD = Multifile(filename, beg = good_start, end = Nimg)
            uid_ = uid + '--fra-%s-%s'%(FD.beg, FD.end)
            print( uid_ )
    
        if run_one_time:    
            t0 = time.time()
            if use_imgsum_norm:
                imgsum_ = imgsum
            else:
                imgsum_ = None            
            g2, lag_steps  = cal_g2p( FD,  ring_mask, bad_frame_list,good_start, num_buf = 8, num_lev= None,
                                    imgsum= imgsum_, norm=norm )
            
            run_time(t0)
            taus = lag_steps * timeperframe
            res_pargs = dict(taus=taus, q_ring_center=q_ring_center, path=data_dir, uid=uid_  ) 
            #save_g2( g2, taus=taus, qr= q_ring_center, qz=None, uid=uid_, path= data_dir )
            g2_pds = save_g2( g2, taus=taus, qr= q_ring_center, qz=None, uid=uid_+'--g2.csv', path= data_dir, return_res=True )
            #if not run_fit_g2:
            #plot_g2( g2, res_pargs= res_pargs,  master_plot='qz',vlim=[0.95, 1.05], geometry='saxs', append_name=  ''  )          
            g2_fit_result, taus_fit, g2_fit = get_g2_fit( g2,  res_pargs=res_pargs, 
                            function = fit_g2_func ,  vlim=[0.95, 1.05], fit_range= None,  
                        fit_variables={'baseline':True, 'beta':True, 'alpha':False,'relaxation_rate':True}, 
                        guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,})  

            res_pargs_fit = dict(taus=taus_fit, q_ring_center= q_ring_center,  
                                     path=data_dir, uid=uid +'_fra-%s-%s'%(FD.beg, FD.end) +'_g2_fit'       )

            g2_fit_paras = save_g2_fit_para_tocsv(g2_fit_result, 
                                filename= 'uid=%s'%uid_  +'_g2_fit_paras.csv', path=data_dir )

            plot_g2( g2, res_pargs= res_pargs, tau_2 = taus_fit, g2_2 = g2_fit,  
                    fit_res= g2_fit_result, function = fit_g2_func, master_plot='qz',    
                   vlim=[0.95, 1.05], geometry='saxs', append_name=  '-fit'  )

            fit_q_rate(  q_ring_center[:],g2_fit_paras['relaxation_rate'], power_variable= False, 
                       uid=uid_  , path= data_dir )

        # For two-time
        data_pixel = None
        if run_two_time:    
            data_pixel =   Get_Pixel_Arrayc( FD, pixelist,  norm=norm ).get_data()
            t0=time.time()    
            g12b = auto_two_Arrayc(  data_pixel,  ring_mask, index = None   )
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

            #max_taus= lag_steps.max()  
            max_taus = Nimg    
            t0=time.time()
            g2b = get_one_time_from_two_time(g12b)[:max_taus]
            run_time(t0)
            tausb = np.arange( g2b.shape[0])[:max_taus] *timeperframe
            res_pargsb = dict(taus=tausb, q_ring_center=q_ring_center,  path=data_dir, uid=uid_       )
            #save_g2( g2b, taus=tausb, qr= q_ring_center, qz=None, uid=uid_ +'-fromTwotime', path= data_dir )
            g2b_pds = save_g2( g2b, taus=tausb, qr= q_ring_center, qz=None, uid=uid_ +'--g2b.csv', path= data_dir, return_res=True )

            #plot_saxs_g2( g2b, tausb,  vlim=[0.95, 1.05], res_pargs=res_pargsb) 


            g2_fit_resultb, taus_fitb, g2_fitb = get_g2_fit( g2b,  res_pargs=res_pargsb, 
                        function = fit_g2_func,  vlim=[0.95, 1.05], fit_range= None,  
                        fit_variables={'baseline':True, 'beta':True, 'alpha':False,'relaxation_rate':True},                                  
                        guess_values={'baseline':1.0,'beta':0.05,'alpha':1.0,'relaxation_rate':0.01,})  

            res_pargs_fitb = dict(taus=taus_fitb, q_ring_center= q_ring_center,  
                                 path=data_dir, uid=uid_  +'_g2b_fit'       )

            g2b_fit_paras = save_g2_fit_para_tocsv(g2_fit_resultb, 
                            filename= 'uid=%s'%uid_  + '_g2b_fit_paras', path=data_dir )


            plot_g2( g2b, res_pargs= res_pargsb, tau_2 = taus_fitb, g2_2 = g2_fitb,  
                fit_res= g2_fit_resultb, function = fit_g2_func, master_plot='qz',    
               vlim=[0.95, 1.05], geometry='saxs', append_name= '-b-fit'  )
            
            if run_two_time and run_one_time:
                plot_saxs_two_g2( g2, taus, 
                         g2b, tausb, res_pargs=res_pargs, vlim=[.95, 1.05], uid= uid_  )

        # Four Time Correlation

        if run_four_time: #have to run one and two first
            t0=time.time()
            g4 = get_four_time_from_two_time(g12b, g2=g2b)[:max_taus]
            run_time(t0)

            taus4 = np.arange( g4.shape[0])*timeperframe
            res_pargs4 = dict(taus=taus4, q_ring_center=q_ring_center, path=data_dir, uid=uid_      )
            #save_saxs_g2(   g4,  res_pargs4, taus=taus4, filename= 'uid=%s'%uid_  + '--g4.csv' )             
            g4_pds = save_g2( g4, taus=taus4, qr= q_ring_center, qz=None, uid=uid_ +'--g4.csv', path= data_dir, return_res=True )
            plot_saxs_g4( g4, taus4,  vlim=[0.95, 1.05], logx=True, res_pargs=res_pargs4) 


        # Speckel Visiblity   
        if run_xsvs:    
            max_cts = get_max_countc(FD, ring_mask )
            qind, pixelist = roi.extract_label_indices(   ring_mask  )
            noqs = len( np.unique(qind) )
            nopr = np.bincount(qind, minlength=(noqs+1))[1:]
            #time_steps = np.array( utils.geometric_series(2,   len(imgs)   ) )
            time_steps = [0,1]  #only run the first two levels
            num_times = len(time_steps)    
            times_xsvs = exposuretime + (2**(  np.arange( len(time_steps) ) ) -1 ) * acquisition_period    
            print( 'The max counts are: %s'%max_cts )

        ### Do historam 
            if ring_avg is  None:
                times_roi, mean_int_sets = cal_each_ring_mean_intensityc(FD, ring_mask, timeperframe = None,  ) 
                ring_avg = np.average( mean_int_sets, axis=0)

            t0=time.time()
            spec_bins, spec_his, spec_std  =  xsvsp( FD, np.int_(ring_mask), norm=None,
                        max_cts=int(max_cts+2),  bad_images=bad_frame_list, only_two_levels=True )    
            spec_kmean =  np.array(  [ring_avg * 2**j for j in  range( spec_his.shape[0] )] )
            run_time(t0)
            run_xsvs_all_lags = False
            if run_xsvs_all_lags:
                times_xsvs = exposuretime +  lag_steps * acquisition_period
                if data_pixel is None:
                    data_pixel =   Get_Pixel_Arrayc( FD, pixelist,  norm=norm ).get_data()       
                t0=time.time()
                spec_bins, spec_his, spec_std, spec_kmean = get_binned_his_std(data_pixel, np.int_(ring_mask), lag_steps )
                run_time(t0) 
            spec_pds =  save_bin_his_std( spec_bins, spec_his, spec_std, filename=uid_+'_spec_res.csv', path=data_dir ) 
            
            #spec_his_pds = save_arrays( spec_his, label= q_ring_center, filename = 'uid=%s--spec_his.csv'%uid_, path=data_dir,return_res=True )
            #spec_std_pds = save_arrays( spec_std, label= q_ring_center, filename = 'uid=%s--spec_std.csv'%uid_, path=data_dir,return_res=True )
            #spec_bins_pds = save_lists( spec_bins, label= q_ring_center, filename = 'uid=%s--spec_bins.csv'%uid_, path=data_dir,return_res=True )
            #np.save(  data_dir + 'uid=%s--spec_his'%uid_, spec_his)
            #np.save(  data_dir + 'uid=%s--spec_std'%uid_, spec_std)
            #np.save(  data_dir + 'uid=%s--K_mean'%uid_, spec_kmean)


            ### Do historam fit by negtive binominal function with maximum likehood method
            qth= None
            #max_bins = 1
            hist_err = spec_std #None# spec_std #None
            ML_val, KL_val,K_ = get_xsvs_fit(  spec_his, spec_kmean,  spec_std = hist_err, max_bins=2,
                                 varyK= False, qth= qth, spec_bins= spec_bins, lag_steps=lag_steps, rois_lowthres= None)


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
                uid=uid_, qth= qth_interest, logy= True, times= times_xsvs, q_ring_center=q_ring_center, path=data_dir)

            plot_xsvs_fit(  spec_his, ML_val, KL_val, K_mean = spec_kmean, spec_std = spec_std,
                          xlim = [0,15], vlim =[.9, 1.1],
                uid=uid_, qth= None, logy= True, times= times_xsvs, q_ring_center=q_ring_center, path=data_dir )

            ### Get contrast
            contrast_factorL = get_contrast( ML_val)
            spec_km_pds = save_KM(  spec_kmean, KL_val, ML_val, qs=q_ring_center, level_time=times_xsvs, uid=uid , path = data_dir )
            #print( spec_km_pds )
    
            plot_g2_contrast( contrast_factorL, g2, times_xsvs, taus, q_ring_center, 
                             vlim=[0.8,1.2], qth = qth_interest, uid=uid_,path = data_dir, legend_size=14)

            plot_g2_contrast( contrast_factorL, g2, times_xsvs, taus, q_ring_center, 
                             vlim=[0.8,1.2], qth = None, uid=uid_,path = data_dir, legend_size=4)
        
        md['mask_file']= mask_path + mask_name
        md['mask'] = mask
        md['NOTEBOOK_FULL_PATH'] = None
        md['good_start'] = good_start
        md['bad_frame_list'] = bad_frame_list
        md['avg_img'] = avg_img
        md['ring_mask'] = ring_mask
        md['q_ring_center']= q_ring_center
        md['q_ring_val'] = q_ring_val
        md['beam_center_x'] =  center[1]
        md['beam_center_y']=  center[0]
        md['beg'] = FD.beg
        md['end'] = FD.end
        md['metadata_file'] = data_dir + 'md.csv-&-md.pkl'
        psave_obj(  md, data_dir + 'uid=%s-md'%uid ) #save the setup parameters
        save_dict_csv( md,  data_dir + 'uid=%s-md.csv'%uid, 'w')

        Exdt = {} 
        for k,v in zip( ['md', 'q_saxs', 'iq_saxs','iqst','qt','roi','avg_img','mask','pixel_mask', 'imgsum','bad_frame_list'], 
                        [md, q_saxs, iq_saxs, iqst, qt,ring_mask,avg_img,mask,pixel_mask, imgsum,bad_frame_list] ):
                         Exdt[ k ] = v
        if run_waterfall:Exdt['wat'] =  wat
        if run_t_ROI_Inten:Exdt['times_roi'] = times_roi;Exdt['mean_int_sets']=mean_int_sets
        if run_one_time:
            for k,v in zip( ['taus','g2','g2_fit_paras'], [taus,g2,g2_fit_paras] ):Exdt[ k ] = v
        if run_two_time:
            for k,v in zip( ['tausb','g2b','g2b_fit_paras', 'g12b'], [tausb,g2b,g2b_fit_paras,g12b] ):Exdt[ k ] = v
        if run_four_time:
            for k,v in zip( ['taus4','g4'], [taus4,g4] ):Exdt[ k ] = v
        if run_xsvs:
            for k,v in zip( ['spec_kmean','spec_pds','times_xsvs','spec_km_pds','contrast_factorL'], 
                           [ spec_kmean,spec_pds,times_xsvs,spec_km_pds,contrast_factorL] ):Exdt[ k ] = v  
                
        export_xpcs_results_to_h5( md['uid'] + '_Res.h5', data_dir, export_dict = Exdt )
        #extract_dict = extract_xpcs_results_from_h5( filename = '%s_Res.h5'% md['uid'], import_dir = data_dir )

        # Creat PDF Report
        pdf_out_dir = os.path.join('/XF11ID/analysis/', CYCLE, username, 'Results/')     
        pdf_filename = "XPCS_Analysis_Report_for_uid=%s%s.pdf"%(uid,pdf_version)
        if run_xsvs:
            pdf_filename = "XPCS_XSVS_Analysis_Report_for_uid=%s%s.pdf"%(uid,pdf_version)
        #pdf_filename
        make_pdf_report( data_dir, uid, pdf_out_dir, pdf_filename, username, 
                        run_fit_form, run_one_time, run_two_time, run_four_time, run_xsvs
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
            for k,v in zip( [ 'q_saxs', 'iq_saxs','iqst','qt','avg_img','mask', 'imgsum','bad_frame_list'], 
                        [ q_saxs, iq_saxs, iqst, qt, avg_img,mask,imgsum,bad_frame_list] ):
                res[ k ] = v
            if run_waterfall:
                res['wat'] =  wat
            if run_t_ROI_Inten:
                res['times_roi'] = times_roi;
                res['mean_int_sets']=mean_int_sets            
            if run_one_time:
                res['g2'] = g2
                res['taus']=taus
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





