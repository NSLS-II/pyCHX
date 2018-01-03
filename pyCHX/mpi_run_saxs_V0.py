#mpiexec -n 4 python mpi_run_saxs_V0.py


from pyCHX.chx_packages import *
from pyCHX.chx_xpcs_xsvs_jupyter import run_xpcs_xsvs_single



from mpi4py import MPI
comm = MPI.COMM_WORLD
print( comm.Get_size(), comm.Get_rank() )




run_pargs=  dict(   
    scat_geometry = 'saxs',
    #scat_geometry = 'gi_saxs',
    
    
    force_compress =  False, #True, #False, #True,#False, 
    para_compress = True,
    run_fit_form = False,
    run_waterfall = True,#False,
    run_t_ROI_Inten = True,
    #run_fit_g2 = True,
    fit_g2_func = 'stretched',
    run_one_time = True,#False,
    run_two_time = True,#False,
    run_four_time = False, #True, #False,
    run_xsvs=True,
    att_pdf_report = True,
    show_plot = False,
                 
    CYCLE = '2016_3', 
    
    #if scat_geometry == 'gi_saxs':
    #mask_path = '/XF11ID/analysis/2016_3/masks/',
    #mask_name =  'Nov16_4M-GiSAXS_mask.npy',
    
    #elif scat_geometry == 'saxs':
    mask_path = '/XF11ID/analysis/2016_3/masks/',
    mask_name = 'Nov28_4M_SAXS_mask.npy',
    
    
    good_start   = 5,     
    #####################################for saxs             
    uniformq = True,
    inner_radius= 0.005,  #0.005 for 50 nmAu/SiO2, 0.006, #for 10nm/coralpor
    outer_radius = 0.04,  #0.04 for 50 nmAu/SiO2, 0.05, #for 10nm/coralpor 
    num_rings = 12,
    gap_ring_number = 6, 
    number_rings= 1, 
    
    ############################for gi_saxs    
    
    #inc_x0 = 1473,
    #inc_y0 = 372,
    #refl_x0 = 1473,
    #refl_y0 = 730,    
    
    qz_start = 0.025,
    qz_end = 0.04,
    qz_num = 3,
    gap_qz_num = 1, 
    #qz_width = ( qz_end - qz_start)/(qz_num +1),

    qr_start =  0.0025,
    qr_end = 0.07,
    qr_num = 14,
    gap_qr_num = 5,     
    
    definde_second_roi = True,
    
    qz_start2 = 0.04,
    qz_end2 = 0.050,
    qz_num2= 1,
    gap_qz_num2 = 1, 
      
    qr_start2 =  0.002,
    qr_end2 = 0.064,
    qr_num2 = 10,
    gap_qr_num2 = 5,    
    
    #qcenters = [ 0.00235,0.00379,0.00508,0.00636,0.00773, 0.00902] #in A-1             
    #width = 0.0002  
    qth_interest = 1, #the intested single qth             
    use_sqnorm = False,    
    use_imgsum_norm = True,
    
    pdf_version = '_1' #for pdf report name    
                )

print( run_pargs ) 

give_uids= True #True
#give_uids= True

if not give_uids:
    start_time, stop_time = '2016-11-29  8:41:00', '2016-11-29  13:15:00'  #4uids    
    start_time, stop_time = '2016-11-30  17:41:00', '2016-11-30  17:46:00' #5uids 
    
    start_time, stop_time = '2016-12-1  14:27:10', '2016-12-1  14:32:10' 
    start_time, stop_time = '2016-12-1  14:36:20', '2016-12-1  14:41:10'
    start_time, stop_time = '2016-12-1  14:50:20', '2016-12-1  14:54:10' #unknown
    start_time, stop_time = '2016-12-1  15:12:20', '2016-12-1  15:16:50' #count : 1 ['c3f523'] (scan num: 10487) (Measurement: sample X1 50nm Au in 7.5% PEG_fast_series_#0 )
    start_time, stop_time = '2016-12-1  15:17:00', '2016-12-1  15:20:30'  #count : 1 ['54c823'] (scan num: 10497) (Measurement: sample X1 50nm Au in 7.5% PEG_slow_series_#0 )
    start_time, stop_time = '2016-12-1  15:21:00', '2016-12-1  15:24:30'  #count : 1 ['07041f'] (scan num: 10507) (Measurement: sample X1 50nm Au in 7.5% PEG_visibility_series_#0 )

    #start_time, stop_time = '2016-12-1  15:33:00', '2016-12-1  15:36:30'    
    #start_time, stop_time = '2016-12-1  15:37:00', '2016-12-1  15:40:30'  
    #start_time, stop_time = '2016-12-1  15:41:00', '2016-12-1  15:44:30'     
    start_time, stop_time = '2016-12-1  15:33:00',  '2016-12-1  15:44:30'  #X3, 10 nm, fast/slow/vs200us
        
    #start_time, stop_time = '2016-12-1  16:05:00', '2016-12-1  16:08:50'    
    #start_time, stop_time = '2016-12-1  16:09:00', '2016-12-1  16:12:30'  
    #start_time, stop_time = '2016-12-1  16:13:00', '2016-12-1  16:16:50'
    start_time, stop_time = '2016-12-1  16:05:00', '2016-12-1  16:16:50' #X3, 10 nm, fast/slow/vs200us

    #start_time, stop_time = '2016-12-1  16:20:00', '2016-12-1  16:25:50'    
    #start_time, stop_time = '2016-12-1  16:26:00', '2016-12-1  16:29:50'  
    #start_time, stop_time = '2016-12-1  16:30:00', '2016-12-1  16:35:50'

    start_time, stop_time = '2016-12-1  16:20:00',  '2016-12-1  16:35:50'

    #X4 2.5%PEG, 10nmAu
    start_time, stop_time = '2016-12-1  17:06:00', '2016-12-1  17:11:50'    
    start_time, stop_time = '2016-12-1  17:12:00', '2016-12-1  17:16:50' 
    start_time, stop_time = '2016-12-1  17:17:00', '2016-12-1  17:22:50'
    start_time, stop_time = '2016-12-1  17:29:00', '2016-12-1  17:33:50'    
    start_time, stop_time = '2016-12-1  17:33:55', '2016-12-1  17:36:56' 
    start_time, stop_time = '2016-12-1  17:37:00', '2016-12-1  17:42:50'

    start_time, stop_time = '2016-12-1  17:06:00',  '2016-12-1  17:42:50'
    
    start_time, stop_time = '2016-12-1  18:03:00',  '2016-12-1  19:12:50'
    start_time, stop_time = '2016-12-1  19:56:00', '2016-12-1  20:11:00' #for X6
    start_time, stop_time = '2016-12-1  20:20', '2016-12-1  20:35:00' #for X5
    start_time, stop_time = '2016-12-1  20:39', '2016-12-1  20:53:00' #for X4, 10 nm
    start_time, stop_time = '2016-12-1  21:17', '2016-12-1  21:31:00' #for X3, 10 nm
    start_time, stop_time = '2016-12-1  21:32', '2016-12-1  21:46:00' #for X2, 50 nm
    start_time, stop_time = '2016-12-1  21:46', '2016-12-1  22:01:00' #for X1, 50 nm
    
    start_time, stop_time = '2016-12-2  11:05', '2016-12-2  11:14:00' #for coralpor, 10,20,40,80,160,320,640
    start_time, stop_time = '2016-12-2  12:03', '2016-12-2  13:28:00' #for coralpor,
 
    start_time, stop_time = '2016-12-1  16:30:00', '2016-12-1  16:31:50' #for 10 nm, 20, for test purpose
    
    start_time, stop_time = '2016-12-1  16:30:00', '2016-12-1  16:31:50' #for 10 nm, 20, for test purpose 
    #start_time, stop_time = '2016-12-01 14:31:31',  '2016-12-01 14:31:40' #for test purpose                            
    sids, uids, fuids = find_uids(start_time, stop_time)
    
    
    
if give_uids:
    uids = [  'b31e61', 'f05d46', '67cfa1', '52cc2d'  ]  
    uids = ['25f88334']
    uids = [ '89297ae8' ]  #for saxs test
    #uid = [ '96c5dd' ] #for gisaxs test
    
#uids =['d83fdb']  #100 frames, for test code
#uids =['1d893a'    ]


def printf( i, s ):
    print('%'*40)
    print ("Processing %i-th-->uid=%s"%(i+1,s) )
    print('%'*40)

t0=time.time()    
    
for i in range( comm.rank, comm.rank+1  ):
    print (i)
    uid = uids[i]
    printf( i, uid  )
    run_xpcs_xsvs_single( uid,  run_pargs )    
    
run_time(t0)
    
    
    
    
    
    
    
    
 
