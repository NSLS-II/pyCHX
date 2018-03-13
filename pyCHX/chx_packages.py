import pickle as cpk
from skimage.draw import line_aa, line, polygon
import historydict


from pyCHX.chx_libs import (np, roi, time, datetime, os, get_events, 
                            getpass, db, get_images,LogNorm, plt,tqdm, utils, Model,
                           multi_tau_lags,  random,  warnings)
from pyCHX.chx_libs import (cmap_vge, cmap_albula, Javascript, EigerHandler, EigerHandler, pims,
                                h5py)


from pyCHX.chx_handlers import use_pims, use_dask
use_pims(db)  #use pims for importing eiger data, register_handler 'AD_EIGER2' and 'AD_EIGER'

from pyCHX.chx_generic_functions import (get_detector, get_fields, get_sid_filenames,  
    load_data, load_mask,get_fields, reverse_updown, ring_edges,get_avg_img,check_shutter_open,
    apply_mask, show_img,check_ROI_intensity,run_time, plot1D, get_each_frame_intensity, 
    create_hot_pixel_mask,show_ROI_on_image,create_time_slice,save_lists, 
    save_arrays, psave_obj,pload_obj, get_non_uniform_edges,
    get_meta_data,   print_dict,    save_dict_csv,  read_dict_csv,
    get_bad_frame_list,  find_bad_pixels,  mask_exclude_badpixel, trans_data_to_pd,
    get_max_countc,find_uids ,    check_bad_uids,   get_averaged_data_from_multi_res,
    get_qval_dict,  save_g2_general, get_g2_fit_general,  plot_g2_general,
    get_q_rate_fit_general, plot_q_rate_fit_general,  save_g2_fit_para_tocsv,  
    update_qval_dict,  update_roi_mask,     combine_images,   create_rectangle_mask, create_cross_mask,  create_polygon_mask, check_lost_metadata, 
    get_fra_num_by_dose,  get_multi_tau_lag_steps, get_series_g2_taus, create_user_folder,
    get_current_pipeline_filename, get_current_pipeline_fullpath,save_current_pipeline,
    filter_roi_mask, mask_badpixels,
    validate_uid,   
    move_beamstop, get_today_date,  get_print_uids, get_last_uids,get_base_all_filenames,create_ring_mask,
    get_image_edge, get_image_with_roi,
    extract_data_from_file, sgolay2d,
    get_roi_nr, get_mass_center_one_roi,  
    get_echos,  pad_length,    save_array_to_tiff,  load_pilatus, ls_dir,
    get_fit_by_two_linear,get_cross_point,get_curve_turning_points, plot_fit_two_linear_fit,linear_fit,
    find_index,  get_detectors,                               
    get_img_from_iq,average_array_withNan,refine_roi_mask,shrink_image,   get_waxs_beam_center,                                     
                                            )
 

from pyCHX.XPCS_SAXS import (get_circular_average,save_lists,get_ring_mask, get_each_ring_mean_intensity,
                             plot_qIq_with_ROI, cal_g2, create_hot_pixel_mask,get_circular_average,get_t_iq, 
                              get_t_iqc,multi_uids_saxs_xpcs_analysis,
                              plot_t_iqc,  plot_circular_average, get_seg_from_ring_mask, 
                                recover_img_from_iq,get_cirucular_average_std,
                                 get_angular_mask, combine_two_roi_mask
                                
                                )


from pyCHX.Two_Time_Correlation_Function import (show_C12, get_one_time_from_two_time, 
                                            get_four_time_from_two_time,rotate_g12q_to_rectangle)
from pyCHX.chx_compress import (combine_binary_files,
                       segment_compress_eigerdata,     create_compress_header,            
                        para_segment_compress_eigerdata,para_compress_eigerdata)

from pyCHX.chx_compress_analysis import ( compress_eigerdata, read_compressed_eigerdata,
                                         Multifile,get_avg_imgc, get_each_frame_intensityc,
            get_each_ring_mean_intensityc, mean_intensityc,cal_waterfallc,plot_waterfallc, 
                    cal_each_ring_mean_intensityc,plot_each_ring_mean_intensityc, get_time_edge_avg_img
)

from pyCHX.SAXS import fit_form_factor,show_saxs_qmap
from pyCHX.chx_correlationc import ( cal_g2c,Get_Pixel_Arrayc,auto_two_Arrayc,get_pixelist_interp_iq,)
from pyCHX.chx_correlationp import (cal_g2p, auto_two_Arrayp,_one_time_process_errorp)

from pyCHX.Create_Report import (create_pdf_report, 
                            create_multi_pdf_reports_for_uids,create_one_pdf_reports_for_uids,
                                    make_pdf_report, export_xpcs_results_to_h5, extract_xpcs_results_from_h5 )

from pyCHX.chx_olog import (LogEntry,Attachment, update_olog_uid, update_olog_id,
                                update_olog_uid_with_file)

from pyCHX.XPCS_GiSAXS import (get_qedge,get_qmap_label,get_qr_tick_label, get_reflected_angles,
convert_gisaxs_pixel_to_q, show_qzr_map, get_1d_qr, get_qzrmap, show_qzr_roi,get_each_box_mean_intensity,
 plot_gisaxs_two_g2,plot_qr_1d_with_ROI,fit_qr_qz_rate,
        multi_uids_gisaxs_xpcs_analysis,plot_gisaxs_g4,get_t_qrc, plot_t_qrc,
                                  get_qzr_map, plot_qzr_map, get_gisaxs_roi, cal_1d_qr, 
                                  get_t_qrc,  plot_qrt_pds )

from pyCHX.chx_specklecp import  ( xsvsc, xsvsp, get_xsvs_fit,plot_xsvs_fit, save_KM,plot_g2_contrast,
                                      get_binned_his_std, get_contrast, save_bin_his_std, get_his_std_from_pds )


from pyCHX.DataGonio import (qphiavg)


