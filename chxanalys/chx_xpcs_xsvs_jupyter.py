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
        run_xsvs = False
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
        inc_x0 = run_pargs['inc_x0']
        inc_y0 = run_pargs['inc_y0']
    except:
        inc_x0 = None
        inc_y0=  None 
        
    if scat_geometry =='saxs' or 'ang_saxs':
        uniformq =      run_pargs['uniformq']
        use_sqnorm = run_pargs['use_sqnorm']
              
        inner_radius= run_pargs['inner_radius']
        outer_radius =   run_pargs['outer_radius']
        gap_ring_number =run_pargs['gap_ring_number']
        num_rings =   run_pargs['num_rings']  
        
        number_rings= run_pargs['number_rings'] 
        qcenters = run_pargs['qcenters']
        
            
        if 'ring_width' in list(run_pargs.keys()):
            ring_width = run_pargs['ring_width']
            if ring_width is None:ring_width =    ( outer_radius - inner_radius)/(num_rings + gap_ring_number)                
        else:
            ring_width =    ( outer_radius - inner_radius)/(num_rings + gap_ring_number)
        
    if scat_geometry =='ang_saxs':  
        width_angle_v =  run_pargs['width_angle_v']#2
        angle_v =  run_pargs['angle_v']#132
        num_angles_v =    run_pargs['num_angles_v']#1  
        width_angle_p =  run_pargs['width_angle_p']#2
        angle_p =  run_pargs['angle_p']#42
        num_angles_p =   run_pargs['num_angles_p']#1
        
    elif scat_geometry =='gi_saxs':
        refl_x0 = run_pargs['refl_x0']
        refl_y0 = run_pargs['refl_y0']        
        qr_start , qr_end, gap_qr_num, qr_num  = run_pargs['qr_start'],run_pargs['qr_end'],run_pargs['gap_qr_num'], run_pargs['qr_num']        
        qz_start , qz_end, gap_qz_num, qz_num  = run_pargs['qz_start'],run_pargs['qz_end'],run_pargs['gap_qr_num'], run_pargs['qz_num']   
        
        if 'qr_width' in list(run_pargs.keys()):
            qr_width = run_pargs['qr_width']
            if qr_width is None:qr_width =   ( qr_end- qr_start)/(qr_num+gap_qr_num) 
        else:
            qr_width =   ( qr_end- qr_start)/(qr_num+gap_qr_num)        
        if 'qz_width' in list(run_pargs.keys()):
            qz_width = run_pargs['qz_width']
            if qz_width is None:qz_width =   ( qz_end- qz_start)/(qz_num+gap_qz_num)
        else:
            qz_width =   ( qz_end- qz_start)/(qz_num+gap_qz_num)

        
        define_second_roi = run_pargs['define_second_roi'] #if True to define another line; else: make it False
        if define_second_roi:
            qr_start2 , qr_end2, gap_qr_num2, qr_num2  = run_pargs['qr_start2'],run_pargs['qr_end2'],run_pargs['gap_qr_num2'], run_pargs['qr_num2']        
            qz_start2 , qz_end2, gap_qz_num2, qz_num2  = run_pargs['qz_start2'],run_pargs['qz_end2'],run_pargs['gap_qr_num2'], run_pargs['qz_num2']   
            
            if 'qr_width2' in list(run_pargs.keys()):
                qr_width2 = run_pargs['qr_width2']
            else:
                qr_width2 =   ( qr_end2- qr_start2)/(qr_num2+gap_qr_num2)        
            if 'qz_width2' in list(run_pargs.keys()):
                qz_width2 = run_pargs['qz_width2']
            else:
                qz_width2 =   ( qz_end2- qz_start2)/(qz_num2+gap_qz_num2)
            
        use_sqnorm = False  #to be developed for gisaxs 
        
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
 
        if scat_geometry =='saxs' or 'ang_saxs':
        
            show_saxs_qmap( avg_img, setup_pargs, width=600,vmin=.1, vmax=np.max(avg_img*.1), logs=True,
               image_name= uidstr + '_img_avg',  save=True)  
            #np.save(  data_dir + 'uid=%s--img-avg'%uid, avg_img)

            hmask = create_hot_pixel_mask( avg_img, threshold = 100, center=center, center_radius= 400)
            qp_saxs, iq_saxs, q_saxs = get_circular_average( avg_img, mask * hmask, pargs=setup_pargs  )
        
            plot_circular_average( qp_saxs, iq_saxs, q_saxs,  pargs=setup_pargs, 
                              xlim=[q_saxs.min(), q_saxs.max()], ylim = [iq_saxs.min(), iq_saxs.max()] )

            pd = trans_data_to_pd( np.where( hmask !=1), 
                    label=[md['uid']+'_hmask'+'x', md['uid']+'_hmask'+'y' ], dtype='list')
            pd.to_csv('/XF11ID/analysis/Commissioning/eiger4M_badpixel.csv', mode='a'  )
            mask =np.array( mask * hmask, dtype=bool) 
            #show_img( mask )
         
            if run_fit_form:
                form_res = fit_form_factor( q_saxs,iq_saxs,  guess_values={'radius': 2500, 'sigma':0.05, 
                 'delta_rho':1E-10 },  fit_range=[0.0001, 0.015], fit_variables={'radius': T, 'sigma':T, 
                 'delta_rho':T},  res_pargs=setup_pargs, xlim=[0.0001, 0.015]) 
            ### Define a non-uniform distributed rings by giving edges
            if not uniformq:    
                #width = 0.0002    
                #number_rings= 1    
                #qcenters = [ 0.00235,0.00379,0.00508,0.00636,0.00773, 0.00902] #in A-1
                edges = get_non_uniform_edges(  qcenters, ring_width, number_rings )    
                inner_radius= None
                outer_radius = None
                ring_width = None
                num_rings = None
            if uniformq:    
                #inner_radius= 0.006  #16
                #outer_radius = 0.05  #112    
                #num_rings = 12                
                edges = None 

            roi_mask, qr, qr_edge = get_ring_mask(  mask, inner_radius=inner_radius, 
                    outer_radius = outer_radius , width = ring_width, num_rings = num_rings, edges=edges,
                                  unit='A',       pargs=setup_pargs   )
            qind, pixelist = roi.extract_label_indices(  roi_mask  ) 
            qr = np.round( qr, 4)
            show_ROI_on_image( avg_img, roi_mask, center, label_on = False, rwidth =700, alpha=.9,  
                             save=True, path=data_dir, uid=uidstr, vmin= np.min(avg_img), vmax= np.max(avg_img) ) 
            qval_dict = get_qval_dict( np.round(qr, 4)  )      

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
        
            if scat_geometry == 'ang_saxs':
                inner_angle_v=  angle_v -  width_angle_v
                outer_angle_v =  angle_v + width_angle_v            
                roi_mask_v, qval_dict_v =  get_seg_from_ring_mask( inner_angle_v, outer_angle_v,
                                                        num_angles_v, width_angle_v, center, roi_mask, qr )            
                inner_angle_p=  angle_p -  width_angle_p
                outer_angle_p =  angle_p + width_angle_p      
                roi_mask_p, qval_dict_p =  get_seg_from_ring_mask( inner_angle_p, outer_angle_p,
                                                        num_angles_p, width_angle_p, center, roi_mask, qr )

        elif scat_geometry == 'gi_saxs':
            
            show_img( avg_img,  vmin=.1, vmax=np.max(avg_img*.1),
                     logs=True, image_name= uidstr + '_img_avg',  save=True, path=data_dir)
            
            alphaf,thetaf, alphai, phi = get_reflected_angles( inc_x0, inc_y0,refl_x0 , refl_y0, Lsd=Ldet )
            qx_map, qy_map, qr_map, qz_map = convert_gisaxs_pixel_to_q( inc_x0, inc_y0,refl_x0,refl_y0, lamda=lambda_, Lsd=Ldet )
            ticks_  = get_qzr_map(  qr_map, qz_map, inc_x0, Nzline=10,  Nrline=10   )
            ticks = ticks_[:4]
            plot_qzr_map(  qr_map, qz_map, inc_x0, ticks = ticks_, data= avg_img, uid= uidstr, path = data_dir   )
            
            Qr = [qr_start , qr_end, qr_width, qr_num]
            Qz=  [qz_start,   qz_end,  qz_width , qz_num ]
            roi_mask, qval_dict = get_gisaxs_roi( Qr, Qz, qr_map, qz_map, mask= mask ) 
            
            if define_second_roi:    
                qval_dict1 = qval_dict.copy()
                roi_mask1 = roi_mask.copy()
                del qval_dict, roi_mask                
                Qr2 = [qr_start2 , qr_end2, qr_width2, qr_num2]
                Qz2=  [qz_start2,   qz_end2,  qz_width2 , qz_num2 ]    
                roi_mask2, qval_dict2 = get_gisaxs_roi( Qr2, Qz2, qr_map, qz_map, mask= mask )
                qval_dict = update_qval_dict(  qval_dict1, qval_dict2 )
                roi_mask = update_roi_mask(  roi_mask1, roi_mask2 )
                
            show_qzr_roi( avg_img, roi_mask, inc_x0, ticks, alpha=0.5, save=True, path=data_dir, uid=uidstr )    
            qind, pixelist = roi.extract_label_indices(roi_mask)
            noqs = len(np.unique(qind))
            nopr = np.bincount(qind, minlength=(noqs+1))[1:]
            
            qr_1d_pds = cal_1d_qr( avg_img, Qr, Qz, qr_map, qz_map, inc_x0,  setup_pargs=setup_pargs )
            plot_qr_1d_with_ROI( qr_1d_pds, qr_center=np.unique( np.array(list( qval_dict.values() ) )[:,0] ),
                    loglog=False, save=True, uid=uidstr, path = data_dir)
            
            Nimg = FD.end - FD.beg 
            time_edge = create_time_slice( N= Nimg, slice_num= 3, slice_width= 1, edges = None )
            time_edge =  np.array( time_edge ) + good_start
            qrt_pds = get_t_qrc( FD, time_edge, Qr, Qz, qr_map, qz_map, path=data_dir, uid = uidstr )    
            plot_qrt_pds( qrt_pds, time_edge, qz_index = 0, uid = uidstr, path =  data_dir )

        ##the below works for all the geometries     
        if scat_geometry !='ang_saxs':    
            roi_inten = check_ROI_intensity( avg_img, roi_mask, ring_number= qth_interest, uid =uidstr, save=True, path=data_dir ) 
        if scat_geometry =='saxs':
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
            g2b = get_one_time_from_two_time(g12b)[:max_taus]
            run_time(t0)
            tausb = np.arange( g2b.shape[0])[:max_taus] *timeperframe     
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
            times_xsvs = exposuretime + (2**(  np.arange( len(time_steps) ) ) -1 ) * acquisition_period    
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

        elif  scat_geometry == 'saxs':
            md['qr']= qr
            md['qr_edge'] = qr_edge
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





