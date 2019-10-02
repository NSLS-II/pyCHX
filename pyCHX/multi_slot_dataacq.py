
def acquisition_from_database(acquisition_database_obid,error_mode='try',focus_correction=False,stop_key='none',OAV_mode='single'):
    """
    #error_mode='try' # 'try' / 'skip': select what do do with slots where the data acquisition has errors: try to collect whatever possible, skip: skip slot alltogether
    """
    
    org_md_keys=list(RE.md.keys())
    obid=acquisition_database_obid
    mount_dict=beamline_pos.find_one({'_id':'mount_dict'})
    #org_md_keys=list(RE.md.keys())
    det_in='unknown'
    data_acq_dict=data_acquisition_collection.find_one({'_id':obid})
    det_list={'500k':'eiger500k','4m':'eiger4m','1m':'eiger1m'}
    for i in data_acq_dict['slots used']:
        acq_count = 0
        if error_mode == 'skip' and data_acq_dict[i]['errors']:
            print (bcolors.FAIL + "Error in data acquisition definition for "+i+" -> no data will be collected for this sample"+ bcolors.ENDC)
        elif error_mode == 'try':
            #try:   ######################################### comment for debugging!!!
                if data_acq_dict[i]['stats'][4] > 0:
                    print(bcolors.WARNING+'\nTAKING DATA FOR '+i+ bcolors.ENDC)
                    acq_count = 1
                    sample_info=samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})
                    # prep data acquisition: move to slot center, taking offsets from beamlin_data
                    offsets=beamline_pos.find_one({'_id':data_acq_dict['sample_mount']+'_sample_center'})['positions']
                    y_off = offsets['diff_yh'];z_off=offsets['diff_zh'];
                    # get correct z-offset for sample holder, depends e.g. on capillary diameter (only capillary and flat cell are implemented)
                    if focus_correction == True:
                        zfocus=get_focus(mount=data_acq_dict['sample_mount'],holder=sample_info['sample']['holder'])
                        print(bcolors.WARNING+'adjusting z-position to keep OAV focused:'+ bcolors.ENDC)
                        print(bcolors.WARNING+'RE(mov(diff.zh,'+str(zfocus)+')) '+ bcolors.ENDC)
                        RE(mov(diff.zh,zfocus))
                    print('center position of '+i+' in sample holder mount '+data_acq_dict['sample_mount']+': '+str(mount_dict[data_acq_dict['sample_mount']][i[4:]]))
                    x_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][0];y_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][1]
                    if not data_acq_dict['sample_mount'] == 'multi':
                        x_off=offsets['diff_xh']
                        print(bcolors.WARNING+'RE(mov(diff.xh,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+')) '+ bcolors.ENDC)
                        RE(mov(diff.xh,-x_cen+x_off,diff.yh,-y_cen+y_off))
                    else:
                        x_off = offsets['sample_x']                          
                        print(bcolors.WARNING+'RE(mov(sample_x,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+',diff.xh,'+str(offsets['diff_xh'])+')) '+ bcolors.ENDC)
                        RE(mov(sample_x,-x_cen+x_off,diff.yh,-y_cen+y_off,diff.xh,offsets['diff_xh']))
                    # prep data acquisition: update md
                    RE.md['automatic data collection for ObjectID']=str(obid)
                    RE.md['sample_mount']=data_acq_dict['sample_mount']
                    RE.md['sample database id']=str(data_acq_dict[i]['sample_id'])
                    RE.md['owner']=sample_info['info']['owner']
                    RE.md['new_spot_method']=sample_info['info']['new_spot_method']
                    md_list=list(sample_info['sample'].keys())
                    for g in md_list:
                        RE.md[g]=sample_info['sample'][g]
                    RE.md['sample']=RE.md['sample name']
                    #print(RE.md)
                    # check what's requested: temp change, wait or data acquisition -> move to next sampling grid point
                    for m in range(len(data_acq_dict[i]['acq_list'])):
                        if not data_acq_dict[i]['acq_completed'][m]: # task in database has not been previously completed
                            if data_acq_dict[i]['acq_list'][m][0] in list(det_list.keys()):   # this is a data series!
                                print(bcolors.WARNING+'next data acquisition: '+str(data_acq_dict[i]['acq_list'][m])+ bcolors.ENDC)
                                if det_in != data_acq_dict[i]['acq_list'][m][0]:   # move to the requested detector
                                    print(bcolors.WARNING+'changing detector to: '+data_acq_dict[i]['acq_list'][m][0]+ bcolors.ENDC)
                                    x_det_pos=beamline_pos.find_one({'_id':data_acq_dict[i]['acq_list'][m][0]+'_in'})['positions']['saxs_detector_x']
                                    y_det_pos=beamline_pos.find_one({'_id':data_acq_dict[i]['acq_list'][m][0]+'_in'})['positions']['saxs_detector_y']
                                    print(bcolors.WARNING+'RE(mov(saxs_detector.x,'+str(x_det_pos)+',saxs_detector.y,'+str(y_det_pos)+')) '+ bcolors.ENDC)
                                    RE(mov(saxs_detector.x,x_det_pos,saxs_detector.y,y_det_pos)) 
                                    det_in=data_acq_dict[i]['acq_list'][m][0]
                                nspm = samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['new_spot_method']
                                if nspm != 'static': # request to change the sample position for each new data point
                                    print('new_spot_method is '+nspm)
                                    if float(get_n_fresh_spots(data_acq_dict[i]['sample_id'])) >= 1: #sufficient number of fresh spots available
                                        print('sufficient number of fresh spots available!')
                                        x_point=np.array(samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['points'][0])
                                        y_point=np.array(samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['points'][1])
                                        dose=np.array(samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['points'][2])
                                        # where is the next sample grid point?
                                        [dose_ind,next_x_point,next_y_point]=next_grid_point(x_point,y_point,dose,mode=nspm)
                                        print('new spot: [x,y]: '+str([next_x_point,next_y_point]))
                    # move to next sample grid point:
                                        x_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][0]+next_x_point;y_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][1]+next_y_point
                                        if not data_acq_dict['sample_mount'] == 'multi':
                                            print(bcolors.WARNING+'RE(mov(diff.xh,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+')) '+ bcolors.ENDC)
                                            RE(mov(diff.xh,-x_cen+x_off,diff.yh,-y_cen+y_off))
                                        else:
                                            x_off = offsets['sample_x']
                                            print(bcolors.WARNING+'RE(mov(sample_x,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+')) '+ bcolors.ENDC)
                                            RE(mov(sample_x,-x_cen+x_off,diff.yh,-y_cen+y_off))
                                        # update dose for sampling grid point and update database:
                                        dose[dose_ind] = data_acq_dict[i]['acq_list'][m][1]*data_acq_dict[i]['acq_list'][m][2]*data_acq_dict[i]['acq_list'][m][4]
                                        fresh_spots = int(sum(dose==0))
                                        update_sample_database_with_new_sampling_grid(data_acq_dict[i]['sample_id'],x_point,y_point,dose,fresh_spots)
                                    else:
                                        print(bcolors.FAIL+'number of fresh spots available is 0! NOT moving to fresh spot.'+bcolors.ENDC)
                                else:
                                    print('new_spot_method is "static": will not change sample spot between data series')
                                # take actual data!!!
                                print(bcolors.WARNING+'att2.set_T('+str(data_acq_dict[i]['acq_list'][m][4])+')'+ bcolors.ENDC)
                                att2.set_T(data_acq_dict[i]['acq_list'][m][4])
                                acql=data_acq_dict[i]['acq_list'][m]
                                print(bcolors.WARNING+'series(det='+det_list[acql[0]]+',expt='+str(acql[1])+',acqp="auto",imnum='+str(acql[2])+',OAV_mode='+OAV_mode+',feedback_on='+str(acql[3])+',comment='+str(acql[1])+'s x'+str(str(acql[2]))+'  '+RE.md['sample']+')' +bcolors.ENDC)
                                series(det=det_list[acql[0]],expt=acql[1],acqp='auto',imnum=acql[2],OAV_mode=OAV_mode,feedback_on=acql[3],comment=str(acql[1])+'s x'+str(str(acql[2]))+'  '+RE.md['sample'])
                                # fake some data acquisition to get a uid:
                                #RE(count([eiger1m_single]))   # this will become series!!
                                uid=db[-1]['start']['uid']
                                #for ics in tqdm(range(100)):
                                #    time.sleep(.1)
                                # add uid to database for compression [can be done by 'series()' in the future]:
                                uid_list=data_acquisition_collection.find_one({'_id':'general_list'})['uid_list']
                                uid_list.append(uid)
                                data_acquisition_collection.update_one({'_id': 'general_list'},{'$set':{'uid_list' : uid_list}})
                                # add uid to sample database:
                                sample_uidlist=samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['uids']
                                sample_uidlist.append(uid)
                                samples_2.update_one({'_id': data_acq_dict[i]['sample_id']},{'$set':{'info.uids' : sample_uidlist}})

                            elif data_acq_dict[i]['acq_list'][m][0] == 'T_ramp':
                                print(bcolors.WARNING+'set_temperature('+str(data_acq_dict[i]['acq_list'][m][1])+',cool_ramp='+str(data_acq_dict[i]['acq_list'][m][2])+',heat_ramp='+str(data_acq_dict[i]['acq_list'][m][2])+')' +bcolors.ENDC)
                                set_temperature(data_acq_dict[i]['acq_list'][m][1],cool_ramp=data_acq_dict[i]['acq_list'][m][2],heat_ramp=data_acq_dict[i]['acq_list'][m][2])
                                print(bcolors.WARNING+'wait_temperature('+str(data_acq_dict[i]['acq_list'][m][3])+')' +bcolors.ENDC)
                                wait_temperature(data_acq_dict[i]['acq_list'][m][3])
                            elif data_acq_dict[i]['acq_list'][m][0] == 'wait':
                                 print(bcolors.WARNING+'RE(sleep('+str(data_acq_dict[i]['acq_list'][m][1])+'))' +bcolors.ENDC)
                                 RE(sleep(data_acq_dict[i]['acq_list'][m][1]))
                            # mark task as complete:
                            acq_completed_list = data_acquisition_collection.find_one({'_id':obid})[i]['acq_completed']
                            acq_completed_list[m] = True
                            data_acquisition_collection.update_one({'_id':obid},{'$set':{i+'.acq_completed' : acq_completed_list}})

                        else:
                            print(bcolors.FAIL+'Task '+str(data_acq_dict[i]['acq_list'][m])+' has been previously completed: skip!'+bcolors.ENDC)

                # clean up: remove metadata
                for q in list(RE.md.keys()):
                    if q not in org_md_keys:
                        waste=RE.md.pop(q)
                else:
                    if acq_count == 0:
                        print(bcolors.WARNING+'SKIP SLOT '+i+': No data points requested'+ bcolors.ENDC)
                    else:
                        print(bcolors.OKGREEN+'\nDATA ACQUISITION FOR SLOT '+i+' COMPLETED.'+ bcolors.ENDC)

            #except:
            #    print (bcolors.FAIL + "Error in data acquisition definition for "+i+" -> no or} not all data collected for this sample"+ bcolors.ENDC) 
    if stop_key != 'none': # key to automatically stop compression and analysis
        uid_list=data_acquisition_collection.find_one({'_id':'general_list'})['uid_list']
        uid_list.append(stop_key)
        data_acquisition_collection.update_one({'_id': 'general_list'},{'$set':{'uid_list' : uid_list}})
        
        
        
        
        
        
        
# data acquisition from database entry:
def acquisition_from_database2(acquisition_database_obid,error_mode='try',focus_correction=False,stop_key='none',OAV_mode='single'):
    """
    #error_mode='try' # 'try' / 'skip': select what do do with slots where the data acquisition has errors: try to collect whatever possible, skip: skip slot alltogether
    """
    
    org_md_keys=list(RE.md.keys())
    obid=acquisition_database_obid
    mount_dict=beamline_pos.find_one({'_id':'mount_dict'})
    org_md_keys=list(RE.md.keys())
    det_in='unknown'
    data_acq_dict=data_acquisition_collection.find_one({'_id':obid})
    det_list={'500k':'eiger500k','4m':'eiger4m','1m':'eiger1m'}
    for i in data_acq_dict['slots used']:
        acq_count = 0
        if error_mode == 'skip' and data_acq_dict[i]['errors']:
            print (bcolors.FAIL + "Error in data acquisition definition for "+i+" -> no data will be collected for this sample"+ bcolors.ENDC)
        elif error_mode == 'try':
            #try:   ######################################### comment for debugging!!!
                if data_acq_dict[i]['stats'][4] > 0:
                    print('\nTAKING DATA FOR '+i)
                    acq_count = 1
                    sample_info=samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})
                    # prep data acquisition: move to slot center, taking offsets from beamlin_data
                    offsets=beamline_pos.find_one({'_id':data_acq_dict['sample_mount']+'_sample_center'})['positions']
                    y_off = offsets['diff_yh'];z_off=offsets['diff_zh'];
                    # get correct z-offset for sample holder, depends e.g. on capillary diameter (only capillary and flat cell are implemented)
                    if focus_correction == True:
                        zfocus=get_focus(mount=data_acq_dict['sample_mount'],holder=sample_info['sample']['holder'])
                        print('adjusting z-position to keep OAV focused:')
                        print(bcolors.WARNING+'RE(mov(diff.zh,'+str(zfocus)+')) '+ bcolors.ENDC)
                    print('center position of '+i+' in sample holder mount '+data_acq_dict['sample_mount']+': '+str(mount_dict[data_acq_dict['sample_mount']][i[4:]]))
                    x_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][0];y_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][1]
                    if not data_acq_dict['sample_mount'] == 'multi':
                        x_off=offsets['diff_xh']
                        print(bcolors.WARNING+'RE(mov(diff.xh,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+')) '+ bcolors.ENDC)
                    else:
                        x_off = offsets['sample_x']                          
                        print(bcolors.WARNING+'RE(mov(sample_x,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+',diff.xh,'+str(offsets['diff_xh'])+')) '+ bcolors.ENDC)
                    # prep data acquisition: update md
                    RE.md['automatic data collection for ObjectID']=str(obid)
                    RE.md['sample_mount']=data_acq_dict['sample_mount']
                    RE.md['sample database id']=str(data_acq_dict[i]['sample_id'])
                    RE.md['owner']=sample_info['info']['owner']
                    RE.md['new_spot_method']=sample_info['info']['new_spot_method']
                    md_list=list(sample_info['sample'].keys())
                    for g in md_list:
                        RE.md[g]=sample_info['sample'][g]
                    RE.md['sample']=RE.md['sample name']
                    #print(RE.md)
                    # check what's requested: temp change, wait or data acquisition -> move to next sampling grid point
                    for m in range(len(data_acq_dict[i]['acq_list'])):
                        if not data_acq_dict[i]['acq_completed'][m]: # task in database has not been previously completed
                            if data_acq_dict[i]['acq_list'][m][0] in list(det_list.keys()):   # this is a data series!
                                print('next data acquisition: '+str(data_acq_dict[i]['acq_list'][m]))
                                if det_in != data_acq_dict[i]['acq_list'][m][0]:   # move to the requested detector
                                    print('changing detector to: '+data_acq_dict[i]['acq_list'][m][0])
                                    x_det_pos=beamline_pos.find_one({'_id':data_acq_dict[i]['acq_list'][m][0]+'_in'})['positions']['saxs_detector_x']
                                    y_det_pos=beamline_pos.find_one({'_id':data_acq_dict[i]['acq_list'][m][0]+'_in'})['positions']['saxs_detector_y']
                                    print(bcolors.WARNING+'RE(mov(det.x,'+str(x_det_pos)+',det.y,'+str(y_det_pos)+')) '+ bcolors.ENDC)
                                    det_in=data_acq_dict[i]['acq_list'][m][0]
                                nspm = samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['new_spot_method']
                                if nspm != 'static': # request to change the sample position for each new data point
                                    print('new_spot_method is '+nspm)
                                    if float(get_n_fresh_spots(data_acq_dict[i]['sample_id'])) >= 1: #sufficient number of fresh spots available
                                        print('sufficient number of fresh spots available!')
                                        x_point=np.array(samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['points'][0])
                                        y_point=np.array(samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['points'][1])
                                        dose=np.array(samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['points'][2])
                                        # where is the next sample grid point?
                                        [dose_ind,next_x_point,next_y_point]=next_grid_point(x_point,y_point,dose,mode=nspm)
                                        print('new spot: [x,y]: '+str([next_x_point,next_y_point]))
                    # move to next sample grid point:
                                        x_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][0]+next_x_point;y_cen=mount_dict[data_acq_dict['sample_mount']][i[4:]][1]+next_y_point
                                        if not data_acq_dict['sample_mount'] == 'multi':
                                            print(bcolors.WARNING+'RE(mov(diff.xh,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+')) '+ bcolors.ENDC)
                                        else:
                                            x_off = offsets['sample_x']
                                            print(bcolors.WARNING+'RE(mov(sample_x,'+str(-x_cen+x_off)+',diff.yh,'+str(-y_cen+y_off)+')) '+ bcolors.ENDC)
                                        # update dose for sampling grid point and update database:
                                        dose[dose_ind] = data_acq_dict[i]['acq_list'][m][1]*data_acq_dict[i]['acq_list'][m][2]*data_acq_dict[i]['acq_list'][m][3]
                                        fresh_spots = int(sum(dose==0))
                                        update_sample_database_with_new_sampling_grid(data_acq_dict[i]['sample_id'],x_point,y_point,dose,fresh_spots)
                                    else:
                                        print(bcolors.FAIL+'number of fresh spots available is 0! NOT moving to fresh spot.'+bcolors.ENDC)
                                else:
                                    print('new_spot_method is "static": will not change sample spot between data series')
                                # take actual data!!!
                                print(bcolors.WARNING+'att2.set_T('+str(data_acq_dict[i]['acq_list'][m][4])+')'+ bcolors.ENDC)
                                acql=data_acq_dict[i]['acq_list'][m]
                                print(bcolors.WARNING+'series(det='+det_list[acql[0]]+',expt='+str(acql[1])+',acqp="auto",imnum='+str(acql[2])+',OAV_mode='+OAV_mode+',feedback_on='+str(acql[3])+',comment='+str(acql[1])+'s x'+str(str(acql[2]))+'  '+RE.md['sample']+')' +bcolors.ENDC)
                                # fake some data acquisition to get a uid:
                                
                                RE(count([det]))
                                
                                uid=db[-1]['start']['uid']
                                for ics in tqdm(range(10)):
                                    time.sleep(1)
                                # add uid to database for compression:
                                uid_list=data_acquisition_collection.find_one({'_id':'general_list'})['uid_list']
                                sample_uidlist=samples_2.find_one({'_id':data_acq_dict[i]['sample_id']})['info']['uids']                                
                                uid_list.append(uid);
                                
                                ##############################################################################
                                #########YG changed here
                                ustr = str(acquisition_database_obid)
                                if ustr not in list( sample_uidlist.keys() ):
                                    sample_uidlist[ ustr ] = []    
                                sample_uidlist[ ustr ].append(  uid ) 
                                
                                data_acquisition_collection.update_one({'_id': 'general_list'},{'$set':{'uid_list' : uid_list}})
                                # add uid to sample database:
                                samples_2.update_one({'_id': data_acq_dict[i]['sample_id']},{'$set':{'info.uids' : sample_uidlist}})
                                ##############################################################################

                            elif data_acq_dict[i]['acq_list'][m][0] == 'T_ramp':
                                print(bcolors.WARNING+'set_temperature('+str(data_acq_dict[i]['acq_list'][m][1])+',cool_ramp='+str(data_acq_dict[i]['acq_list'][m][2])+',heat_ramp='+str(data_acq_dict[i]['acq_list'][m][2])+')' +bcolors.ENDC)
                                print(bcolors.WARNING+'wait_temperature('+str(data_acq_dict[i]['acq_list'][m][3])+')' +bcolors.ENDC)      
                            elif data_acq_dict[i]['acq_list'][m][0] == 'wait':
                                 print(bcolors.WARNING+'RE(sleep('+str(data_acq_dict[i]['acq_list'][m][1])+'))' +bcolors.ENDC) 
                            # mark task as complete:
                            acq_completed_list = data_acquisition_collection.find_one({'_id':obid})[i]['acq_completed']
                            acq_completed_list[m] = True
                            data_acquisition_collection.update_one({'_id':obid},{'$set':{i+'.acq_completed' : acq_completed_list}})

                        else:
                            print('Task '+str(data_acq_dict[i]['acq_list'][m])+' has been previously completed: skip!')

                # clean up: remove metadata
                for q in list(RE.md.keys()):
                    if q not in org_md_keys:
                        waste=RE.md.pop(q)
                else:
                    if acq_count == 0:
                        print(bcolors.WARNING+'SKIP SLOT '+i+': No data points requested'+ bcolors.ENDC)
                    else:
                        print('\nDATA ACQUISITION FOR SLOT '+i+' COMPLETED.')

            #except:
            #    print (bcolors.FAIL + "Error in data acquisition definition for "+i+" -> no or} not all data collected for this sample"+ bcolors.ENDC) 
    if stop_key != 'none': # key to automatically stop compression and analysis
        uid_list=data_acquisition_collection.find_one({'_id':'general_list'})['uid_list']
        uid_list.append(stop_key)
        data_acquisition_collection.update_one({'_id': 'general_list'},{'$set':{'uid_list' : uid_list}})
        













class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
print (bcolors.WARNING + "Warning: No active frommets remain. Continue?" 
      + bcolors.ENDC)


def single_slot_report(multi_sample,slot_key,T_start=24):
    slot1_list=[];error_trac=False
    data_time=0;temp_time=0;wait_time=0;data_points=0
    if multi_sample[slot_key]['data_series'] != 'none':
        for i in list(multi_sample[slot_key]['data_acq_seq'].keys()): # all entries in sequence
            if isinstance(multi_sample[slot_key]['data_acq_seq'][i][0], int): # only look at eiger series
                data_points=data_points+multi_sample[slot_key]['data_acq_seq'][i][0]*len(list(multi_sample[slot_key]['data_series'].keys()))
                for k in range(multi_sample[slot_key]['data_acq_seq'][i][0]):   # 2 repeats
                    #print(len(multi_sample[slot_key]['data_acq_seq'][i][1:]))
                    for l in range(len(multi_sample[slot_key]['data_acq_seq'][i][1:])):     # number of different data acquisitions
                        #print(multi_sample[slot_key]['data_acq_seq'][i][l+1]) # print data acquisitions
                        #print(multi_sample[slot_key]['data_series'][multi_sample['slot1']['data_acq_seq'][i][l+1]])
                        slot1_list.append(multi_sample[slot_key]['data_series'][multi_sample[slot_key]['data_acq_seq'][i][l+1]])
                        data_time=data_time+multi_sample[slot_key]['data_series'][multi_sample[slot_key]['data_acq_seq'][i][l+1]][1]*multi_sample[slot_key]['data_series'][multi_sample[slot_key]['data_acq_seq'][i][l+1]][2]
            elif multi_sample[slot_key]['data_acq_seq'][i][0]=='T_ramp':
                slot1_list.append(multi_sample[slot_key]['data_acq_seq'][i])
                if multi_sample[slot_key]['data_acq_seq'][i][2] !=0:
                    ramp_speed = multi_sample[slot_key]['data_acq_seq'][i][2]/60
                else: ramp_speed = 9E10
                Temp_time=np.abs(T_start-multi_sample[slot_key]['data_acq_seq'][i][1])/ramp_speed+2*multi_sample[slot_key]['data_acq_seq'][i][3]
                T_start=multi_sample[slot_key]['data_acq_seq'][i][1]
                #print('time for temperature change: '+str(Temp_time)+' new starting temperature: '+str(T_start))
                temp_time=temp_time+Temp_time
            elif multi_sample[slot_key]['data_acq_seq'][i][0]=='wait':
                slot1_list.append(multi_sample[slot_key]['data_acq_seq'][i])
                #print('wait time: '+str(multi_sample[slot_key]['data_acq_seq'][i][1]))
                wait_time=wait_time+multi_sample[slot_key]['data_acq_seq'][i][1]
        #print('# data points: '+str(data_points))
    slot1_stats=[data_time,temp_time,wait_time,T_start,data_points]
    #print('Report for: '+slot_key+'\n Sample information:')
    print (bcolors.FAIL + bcolors.BOLD + '\nReport for: '+slot_key+ bcolors.ENDC+'\n Sample information:')
    print(samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['sample'])
    print('other information:')
    d={}
    for k in ['owner','date entered','new_spot_method']:
        d[k]=samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info'][k]
    print(d),print('\n')
    if multi_sample[slot_key]['data_series'] != 'none':
        nump=get_n_fresh_spots(multi_sample[slot_key]['sample_id'])
        if  not samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info']['new_spot_method'] in ['random','static','consecutive','from_center']:
            print (bcolors.FAIL + "Error: no method defined to move to fresh sample spots during data acquisition."+ bcolors.ENDC)
            print("Call add_new_spot_method(ObjectId('"+str(multi_sample[slot_key]['sample_id'])+"'),interactive=True) to define a sampling grid for this sample.")
            error_trac=True
        if nump != False and nump >= data_points:
            print('number of fresh sample spots available: '+str(nump)+' data points: '+str(data_points)+' -> OK!')
        elif nump != False and nump < data_points and samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info']['new_spot_method'] != 'static':
            print (bcolors.FAIL + "Error: not enough fresh spots available for data acquisition.\n Spots available: "+str(nump)+" spots required: "+str(data_points)+ bcolors.ENDC)
            print("Call add_sampling_grid(ObjectId('"+str(multi_sample[slot_key]['sample_id'])+"'),interactive=True) to define a new sampling grid for this sample.")
            error_trac=True
        elif nump == False and samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info']['new_spot_method'] != 'static':
            print (bcolors.FAIL + "Error: no fresh spots available for data acquisition. Sampling grid not defined. Need at least "+str(data_points)+" points to execute selected data acquistion sequence."+ bcolors.ENDC)
            print("Call add_sampling_grid(ObjectId('"+str(multi_sample[slot_key]['sample_id'])+"'),interactive=True) to define a sampling grid for this sample.")
            error_trac=True
        print('\ndata acquistion series defined for this slot: ')
        for i in list(multi_sample[slot_key]['data_series'].keys()):
            d=multi_sample[slot_key]['data_series'][i][1]*multi_sample[slot_key]['data_series'][i][2]*multi_sample[slot_key]['data_series'][i][4]
            tau_min=multi_sample[slot_key]['data_series'][i][1];tau_max=multi_sample[slot_key]['data_series'][i][1]*multi_sample[slot_key]['data_series'][i][2]
            print(i,[multi_sample[slot_key]['data_series'][i]],' -> dose: '+str(d)+'s   taus=['+str([tau_min, tau_max])+']s')
        print('\nacquisition sequence defined for this slot:')
        for i in list(multi_sample[slot_key]['data_acq_seq'].keys()):
            print(i,multi_sample[slot_key]['data_acq_seq'][i])
        print('\ntime for data acquistion [s]: '+str(data_time)+'\ntime for temperature ramps [s]: '+str(temp_time)+'\nwait time [s]: '+str(wait_time)+'\ntotal time for '+slot_key+' [hh:mm:ss]: '+time.strftime("%H:%M:%S", time.gmtime(wait_time+temp_time+data_time)))
        print('starting temperature for next slot: '+str(T_start)+'C\nnumber of data points to be collected: '+str(data_points))
    if multi_sample[slot_key]['data_series'] == 'none' or multi_sample[slot_key]['data_acq_seq'] == 'none': 
        print(bcolors.WARNING + "Warning: no data acquisition defined for this slot."+ bcolors.ENDC)
    return [slot1_list,slot1_stats,error_trac]




def single_slot_report2(multi_sample,slot_key,T_start=24):
    slot1_list=[];error_trac=False
    data_time=0;temp_time=0;wait_time=0;data_points=0
    if multi_sample[slot_key]['data_series'] != 'none':
        for i in list(multi_sample[slot_key]['data_acq_seq'].keys()): # all entries in sequence
            if isinstance(multi_sample[slot_key]['data_acq_seq'][i][0], int): # only look at eiger series
                data_points=data_points+multi_sample[slot_key]['data_acq_seq'][i][0]*len(list(multi_sample[slot_key]['data_series'].keys()))
                for k in range(multi_sample[slot_key]['data_acq_seq'][i][0]):   # 2 repeats
                    #print(len(multi_sample[slot_key]['data_acq_seq'][i][1:]))
                    for l in range(len(multi_sample[slot_key]['data_acq_seq'][i][1:])):     # number of different data acquisitions
                        #print(multi_sample[slot_key]['data_acq_seq'][i][l+1]) # print data acquisitions
                        #print(multi_sample[slot_key]['data_series'][multi_sample['slot1']['data_acq_seq'][i][l+1]])
                        slot1_list.append(multi_sample[slot_key]['data_series'][multi_sample[slot_key]['data_acq_seq'][i][l+1]])
                        data_time=data_time+multi_sample[slot_key]['data_series'][multi_sample[slot_key]['data_acq_seq'][i][l+1]][1]*multi_sample[slot_key]['data_series'][multi_sample[slot_key]['data_acq_seq'][i][l+1]][2]
            elif multi_sample[slot_key]['data_acq_seq'][i][0]=='T_ramp':
                slot1_list.append(multi_sample[slot_key]['data_acq_seq'][i])
                if multi_sample[slot_key]['data_acq_seq'][i][2] !=0:
                    ramp_speed = multi_sample[slot_key]['data_acq_seq'][i][2]/60
                else: ramp_speed = 9E10
                Temp_time=np.abs(T_start-multi_sample[slot_key]['data_acq_seq'][i][1])/ramp_speed+2*multi_sample[slot_key]['data_acq_seq'][i][3]
                T_start=multi_sample[slot_key]['data_acq_seq'][i][1]
                #print('time for temperature change: '+str(Temp_time)+' new starting temperature: '+str(T_start))
                temp_time=temp_time+Temp_time
            elif multi_sample[slot_key]['data_acq_seq'][i][0]=='wait':
                slot1_list.append(multi_sample[slot_key]['data_acq_seq'][i])
                #print('wait time: '+str(multi_sample[slot_key]['data_acq_seq'][i][1]))
                wait_time=wait_time+multi_sample[slot_key]['data_acq_seq'][i][1]
        #print('# data points: '+str(data_points))
    slot1_stats=[data_time,temp_time,wait_time,T_start,data_points]
    #print('Report for: '+slot_key+'\n Sample information:')
    ss = ''
    sx =  bcolors.FAIL + bcolors.BOLD + '\nReport for: '+slot_key+ bcolors.ENDC+'\n Sample information:'
    sx1 =    '\nReport for: '+slot_key+ '\n Sample information:'    
    print (sx)
    ss += sx1
    sx = str(samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['sample'])
    print(sx)
    ss += '\n'
    ss += sx
    ss += '\n'
    sx = 'other information:'
    print(sx)
    ss += sx
    d={}
    dd = {}
    for k in ['owner','date entered','new_spot_method']:
        d[k]=samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info'][k]
        dd[k] =    str(d[k])         
    #sx = '\n' + str(d)
    sx =  '\n' + str(dd)
    print(  str(d) )
    ss += sx
    sx =  '\n'
    print(sx)
    ss += sx    
    #print(d),print('\n')
    if multi_sample[slot_key]['data_series'] != 'none':
        nump=get_n_fresh_spots(multi_sample[slot_key]['sample_id'])
        if  not samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info']['new_spot_method'] in ['random','static','consecutive','from_center']:
            sx =  bcolors.FAIL + "Error: no method defined to move to fresh sample spots during data acquisition."+ bcolors.ENDC
            print (sx)
            ss += sx
            sx = "Call add_new_spot_method(ObjectId('"+str(multi_sample[slot_key]['sample_id'])+"'),interactive=True) to define a sampling grid for this sample."
            print(sx)
            ss += sx
            error_trac=True
        if nump != False and nump >= data_points:
            sx = 'number of fresh sample spots available: '+str(nump)+' data points: '+str(data_points)+' -> OK!'
            print(sx)
            ss = ss + '\n' +  sx + '\n'
             
        elif nump != False and nump < data_points and samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info']['new_spot_method'] != 'static':
            sx = bcolors.FAIL + "Error: not enough fresh spots available for data acquisition.\n Spots available: "+str(nump)+" spots required: "+str(data_points)+ bcolors.ENDC
            print(sx)
            ss += sx
            sx = "Call add_sampling_grid(ObjectId('"+str(multi_sample[slot_key]['sample_id'])+"'),interactive=True) to define a new sampling grid for this sample."
            print(sx)
            ss += sx           
            error_trac=True
        elif nump == False and samples_2.find_one({'_id':multi_sample[slot_key]['sample_id']})['info']['new_spot_method'] != 'static':
 
            sx = bcolors.FAIL + "Error: no fresh spots available for data acquisition. Sampling grid not defined. Need at least "+str(data_points)+" points to execute selected data acquistion sequence."+ bcolors.ENDC
            print(sx)
            ss += sx    
            sx = "Call add_sampling_grid(ObjectId('"+str(multi_sample[slot_key]['sample_id'])+"'),interactive=True) to define a sampling grid for this sample."
            print(sx)
            ss += sx            
            error_trac=True
        sx = '\ndata acquistion series defined for this slot: '
        print(sx)
        ss = ss  +  sx + '\n'
        
        for i in list(multi_sample[slot_key]['data_series'].keys()):
            d=multi_sample[slot_key]['data_series'][i][1]*multi_sample[slot_key]['data_series'][i][2]*multi_sample[slot_key]['data_series'][i][3]
            tau_min=multi_sample[slot_key]['data_series'][i][1];tau_max=multi_sample[slot_key]['data_series'][i][1]*multi_sample[slot_key]['data_series'][i][2]
            sx = (i,[multi_sample[slot_key]['data_series'][i]],' -> dose: '+str(d)+'s   taus=['+str([tau_min, tau_max])+']s')
            sx =  str(sx) 
            print(sx)
            ss = ss  +  sx + '\n'
        sx = ('\nacquisition sequence defined for this slot:')
        print(sx)
        ss = ss  +  sx + '\n'        
        for i in list(multi_sample[slot_key]['data_acq_seq'].keys()):
            sx = (i,multi_sample[slot_key]['data_acq_seq'][i])
            sx =  str(sx) 
            print(sx)
            ss = ss  +  sx + '\n'          
        sx = ('\ntime for data acquistion [s]: '+str(data_time)+'\ntime for temperature ramps [s]: '+str(temp_time)+'\nwait time [s]: '+str(wait_time)+'\ntotal time for '+slot_key+' [hh:mm:ss]: '+time.strftime("%H:%M:%S", time.gmtime(wait_time+temp_time+data_time)))
        print(sx)
        ss = ss  +  sx + '\n'        
        sx = ('starting temperature for next slot: '+str(T_start)+'C\nnumber of data points to be collected: '+str(data_points))
        print(sx)
        ss = ss  +  sx + '\n'  
    if multi_sample[slot_key]['data_series'] == 'none' or multi_sample[slot_key]['data_acq_seq'] == 'none': 
        sx = (bcolors.WARNING + "Warning: no data acquisition defined for this slot."+ bcolors.ENDC)
        print(sx)
        ss = ss  +  sx + '\n'  
    #print(ss)
    return [slot1_list,slot1_stats,error_trac], ss



def multi_slot_report2(multi_sample,T_start=24, png_name = None):
    acq_dict={}   # to become the database for data acquisition macro
    population={} # dictionary for sample mount visualization
    glob_error=[]
    total_time=0
    population['mount']=multi_sample['sample_mount']
    slots_used=list(multi_sample.keys())
    for i in ['sample_mount','date entered','timestamp','owner']:
        slots_used.remove(i)
        acq_dict[i]=multi_sample[i]
    acq_dict['slots used']=slots_used 
    ss = ''
    s0 = 'Report for sample dictionary: multi_sample. Slots_used: '+str(slots_used)
    print(s0)
    ss = ss + s0 + '\n'
    for s in slots_used:
        sam_dict=samples_2.find_one({'_id':multi_sample[s]['sample_id']})
        population[s[4:]]=[sam_dict['sample']['holder'][0],sam_dict['sample']['label']]
        [slot1_list,slot1_stats,error_trac], sx = single_slot_report2(multi_sample,s,T_start=T_start)
        ss += sx
        total_time=total_time+sum(slot1_stats[0:3])
        T_start=slot1_stats[3]
        acq_dict[s]={'acq_list':slot1_list,'acq_completed':[False]*np.shape(slot1_list)[0],'stats':slot1_stats,'errors':error_trac,'sample_id':multi_sample[s]['sample_id']}
        glob_error.append(error_trac)
        #print(slot1_list)
        #print(np.shape(slot1_list))
        #print(slot1_stats)
        #print(error_trac)
    s1 =  bcolors.FAIL + bcolors.BOLD + '\nSetup contains a total of '+str(sum(glob_error))+' errors! No data will be collected for slots with errors.'+ bcolors.ENDC   
    s2 = '\ntotal time for data collection, temperature ramps and waiting [hh:mm:ss]: '+time.strftime("%H:%M:%S", time.gmtime(total_time))
    print ( s1 )    
    print(s2)
    s3 = '\n\nYour sample mount should look like this:'
    print(s3)
    ss += s1
    ss += s2
    ss += s3    
    draw_sample_mounts(population, png_name = png_name)
    return acq_dict, ss



def multi_slot_report(multi_sample,T_start=24, png_name = None):
    acq_dict={}   # to become the database for data acquisition macro
    population={} # dictionary for sample mount visualization
    glob_error=[]
    total_time=0
    population['mount']=multi_sample['sample_mount']
    slots_used=list(multi_sample.keys())
    for i in ['sample_mount','date entered','timestamp','owner']:
        slots_used.remove(i)
        acq_dict[i]=multi_sample[i]
    acq_dict['slots used']=slots_used   
    print('Report for sample dictionary: multi_sample. Slots_used: '+str(slots_used))
    for s in slots_used:
        sam_dict=samples_2.find_one({'_id':multi_sample[s]['sample_id']})
        population[s[4:]]=[sam_dict['sample']['holder'][0],sam_dict['sample']['label']]
        [slot1_list,slot1_stats,error_trac]=single_slot_report(multi_sample,s,T_start=T_start)
        total_time=total_time+sum(slot1_stats[0:3])
        T_start=slot1_stats[3]
        acq_dict[s]={'acq_list':slot1_list,'acq_completed':[False]*np.shape(slot1_list)[0],'stats':slot1_stats,'errors':error_trac,'sample_id':multi_sample[s]['sample_id']}
        glob_error.append(error_trac)
        #print(slot1_list)
        #print(np.shape(slot1_list))
        #print(slot1_stats)
        #print(error_trac)
    print (bcolors.FAIL + bcolors.BOLD + '\nSetup contains a total of '+str(sum(glob_error))+' errors! No data will be collected for slots with errors.'+ bcolors.ENDC)
    print('\ntotal time for data collection, temperature ramps and waiting [hh:mm:ss]: '+time.strftime("%H:%M:%S", time.gmtime(total_time)))
    print('\n\nYour sample mount should look like this:')
    draw_sample_mounts(population, png_name = png_name)
    return acq_dict

def data_acquisition_dictionary_to_database(dictionary):
    """
    moves data acquisition dictionary 'dictionary' created by multi_slot_report() to data acquistion database and returns database key for this entry
    function is interactive: user input to add/not add dictionary to database
    Function checks for Errors in input dictionary, however, it will NOT prevent to add a faulty dictionary to the database.
    by LW 09/03/2018
    """
    # check for errors and issue warning:
    glob_error=[]
    for s in dictionary['slots used']:
        glob_error.append(dictionary[s]['errors'])
    if sum(glob_error) > 0:
        print(bcolors.FAIL + bcolors.BOLD + '\n Data acquisition dictionary contains errors! No data will be collected for slots with errors!'+ bcolors.ENDC)
    else:
        user_input = input('Add dictonary to database for data acquisition? yes/no: ')
        if user_input != 'yes':
            print('Nothing added to database...')
        else: 
            x = data_acquisition_collection.insert_one(dictionary)
            if x.acknowledged == True:
                print('\nInformation successfully added to data acquisition database!')
                print('New database key added: '+str(x.inserted_id))
                return x.inserted_id
            else: print(bcolors.FAIL + bcolors.BOLD + '\n Error: something went wrong...data NOT added to database.'+ bcolors.ENDC) 

# sample holder/mount visualization sample holder

# define sample mounts -> could be hard-coded in startup file...
mount_dict={
    'multi': {'1':[-115.5,0],'2':[-115.5+16.5,0],'3':[-115.5+2*16.5,0],'4':[-115.5+3*16.5,0],'5':[-115.5+4*16.5,0],'6':[-115.5+5*16.5,0],'7':[-115.5+6*16.5,0],
                '8':[-115.5+7*16.5,0],'9':[-115.5+8*16.5,0],'10':[-115.5+9*16.5,0],'11':[-115.5+10*16.5,0],'12':[-115.5+11*16.5,0],'13':[-115.5+12*16.5,0],
                '14':[-115.5+13*16.5,0],'15':[-115.5+14*16.5,0]},
    'flat_Tcell': {'1':[0,0]},
    'inair_flat_cell':{'1':[0,0]},
    'wafer_Tcell': {'1':[0,0]},
    'capillary_Tcell':{'1':[0,-4.5],'2':[0,0],'3':[0,4.5],},
    'inair_3capillary':{'1':[-5,0],'2':[0,0],'3':[5,0],},
    
}

def draw_sample_mounts(population, png_name=None):
    if population['mount'] == 'multi':
        slots={'1':[-115.5,0],'2':[-115.5+16.5,0],'3':[-115.5+2*16.5,0],'4':[-115.5+3*16.5,0],'5':[-115.5+4*16.5,0],'6':[-115.5+5*16.5,0],'7':[-115.5+6*16.5,0],
                '8':[-115.5+7*16.5,0],'9':[-115.5+8*16.5,0],'10':[-115.5+9*16.5,0],'11':[-115.5+10*16.5,0],'12':[-115.5+11*16.5,0],'13':[-115.5+12*16.5,0],
                '14':[-115.5+13*16.5,0],'15':[-115.5+14*16.5,0]}
        mount_dims = [-125,25]
        plt.subplots(nrows=1, ncols=1,figsize=(24, 4))
        #f = plt.gcf()
        plt.plot([-120,-120,120,120,-120],[-6,6,6,-6,-6],'k--',linewidth=3, zorder=1)
        used_slots = list(set(population.keys()).intersection(list(slots.keys())))
        unused_slots=set(list(slots.keys()))-set(used_slots)
        for i in used_slots:
            if population[i][0] == 'flat_cell':
                draw_flatcell(slots[i][0],slots[i][1],mount_dims)
            elif population[i][0] == 'capillary':
                draw_cap_holder(slots[i][0],slots[i][1],mount_dims)
            elif population[i][0] == 'wafer':
                draw_waf_holder(slots[i][0],slots[i][1],mount_dims)
            elif population[i][0] == 'alignment_flat':
                draw_alignment_flat(slots[i][0],slots[i][1],mount_dims)
            #holder = mpatches.Rectangle([slots[i][0]-7,slots[i][1]-13,], 14, 26, angle=0.0, color = 'gray', capstyle = 'round',joinstyle='bevel')
            #ax = plt.gca()
            #ax.add_artist(holder)
            #plt.plot(str(slots[i][0]),str(slots[i][1]),'ks')
            plt.text(slots[i][0]+6, -20, 'slot #'+i+'\n'+population[i][1][:10]+'\n'+population[i][1][10:20], bbox=dict(facecolor='red', alpha=0.5))
        for i in unused_slots:
            draw_empty_flat_slot(slots[i][0],slots[i][1],mount_dims)
            plt.text(slots[i][0]+6, -20, 'slot #'+i+'\nlab: ', bbox=dict(facecolor='white', alpha=0.5))
        ax = plt.gca()
        ax.set_xlim(125,-125)
        ax.set_ylim(-25,20)
        #plt.show()
    elif population['mount'] == 'flat_Tcell':
        slots={'1':[0,0]}
        mount_dims = [-25,17]
        plt.subplots(nrows=1, ncols=1,figsize=(8, 4))
        for i in slots.keys():         
            draw_flat_Tcell(slots[i][0],slots[i][1],mount_dims)
            plt.text(24,12, 'slot #'+i+'\nsample: '+population[i][1], bbox=dict(facecolor='red', alpha=0.5))
    elif population['mount'] == 'inair_flat_cell':
        slots={'1':[0,0]}
        mount_dims = [-25,17]
        plt.subplots(nrows=1, ncols=1,figsize=(8, 4))
        for i in slots.keys():         
            draw_inair_flat_cell(slots[i][0],slots[i][1],mount_dims)
            plt.text(24,12, 'slot #'+i+'\nsample: '+population[i][1], bbox=dict(facecolor='red', alpha=0.5))
    elif population['mount'] == 'wafer_Tcell':
        slots={'1':[0,0]}
        mount_dims = [-25,17]
        plt.subplots(nrows=1, ncols=1,figsize=(8, 4))
        for i in slots.keys():         
            draw_wafer_Tcell(slots[i][0],slots[i][1],mount_dims)
            plt.text(24,12, 'slot #'+i+'\nsample: '+population[i][1], bbox=dict(facecolor='red', alpha=0.5))
    elif population['mount'] == 'capillary_Tcell':
        slots={'1':[0,-4.5],'2':[0,0],'3':[0,4.5],}
        mount_dims = [-30,17]
        #x_pos=slots[i][0];y_pos=slots[i][1]
        used_slots = list(set(population.keys()).intersection(list(slots.keys())))
        unused_slots=set(list(slots.keys()))-set(used_slots)
        plt.subplots(nrows=1, ncols=1,figsize=(8, 4))
        holder = mpatches.Rectangle([0-13,-7], 26, 14, angle=0.0, color = 'gray')
        ax = plt.gca()
        ax.add_artist(holder)
        holder = mpatches.Rectangle([0-13,0-10], 30, 3, angle=0.0, color = 'lightgray')
        ax = plt.gca()
        ax.add_artist(holder)
        holder = mpatches.Rectangle([0-16,0-15], 3, 30, angle=0.0, color = 'lightgray')
        ax = plt.gca()
        ax.add_artist(holder)
        ax.set_xlim(-mount_dims[0],mount_dims[0])
        ax.set_ylim(-mount_dims[1],mount_dims[1])
        for i in used_slots:
            draw_horz_capillary(slots[i][0],slots[i][1],mount_dims)
            plt.text(slots[i][0]+29,slots[i][1], 'slot #'+i+'\nsamp.: '+population[i][1], bbox=dict(facecolor='red', alpha=0.5))
        for i in unused_slots:
            x_pos=slots[i][0];y_pos=slots[i][1]
            holder = mpatches.Rectangle([x_pos-13,y_pos-1], 25, 2, angle=0.0,edgecolor = 'k',linewidth = 2, linestyle='--', facecolor = 'none',zorder=2)
            ax = plt.gca()
            ax.add_artist(holder)
            ax.set_xlim(-mount_dims[0],mount_dims[0])
            ax.set_ylim(-mount_dims[1],mount_dims[1])
            plt.text(slots[i][0]+29, y_pos, 'slot #'+i+'\nlab: ', bbox=dict(facecolor='white', alpha=0.5))
        #plt.show()
    elif population['mount'] == 'inair_3capillary':
        slots={'1':[-5,0],'2':[0,0],'3':[5,0],}
        mount_dims = [-20,25]
        plt.subplots(nrows=1, ncols=1,figsize=(7, 4))
        holder = mpatches.Rectangle([-14,-12], 28, 17, angle=0.0, color = 'gray',zorder=3)
        ax = plt.gca()
        ax.add_artist(holder)
        holder = mpatches.Circle([0,0], 15, color = 'lightgray',zorder=1)
        ax = plt.gca()
        ax.add_artist(holder)
        holder = mpatches.Circle([0,0], 10, color = 'white',zorder=2)
        ax = plt.gca()
        ax.add_artist(holder)
        holder = mpatches.Rectangle([-9,-7], 18, 10, angle=0.0, color = 'white',zorder=4)
        ax = plt.gca()
        ax.add_artist(holder)
        ax.set_xlim(-mount_dims[0],mount_dims[0])
        ax.set_ylim(-mount_dims[1],mount_dims[1])
        #x_pos=slots[i][0];y_pos=slots[i][1]
        used_slots = list(set(population.keys()).intersection(list(slots.keys())))
        unused_slots=set(list(slots.keys()))-set(used_slots)
        for i in used_slots:
            draw_vert_capillary(slots[i][0],slots[i][1],mount_dims)
            plt.text(slots[i][0]+2.5,slots[i][1]+16, 'slot #'+i+'\n'+population[i][1][:5]+'\n'+population[i][1][5:15], bbox=dict(facecolor='red', alpha=0.5))
        for i in unused_slots:
            x_pos=slots[i][0];y_pos=slots[i][1]
            holder = mpatches.Rectangle([x_pos-1,y_pos-10], 2, 21, angle=0.0,edgecolor = 'k',linewidth = 2, linestyle='--', facecolor = 'none',zorder=8)
            ax = plt.gca()
            ax.add_artist(holder)
            ax.set_xlim(-mount_dims[0],mount_dims[0])
            ax.set_ylim(-mount_dims[1],mount_dims[1])
            plt.text(slots[i][0]+2.5,slots[i][1]+16, 'slot #'+i+'\nlab: ', bbox=dict(facecolor='white', alpha=0.5))
    if png_name is None:
        png_name = 'Data_Acquistion_Plot_%s.png'%(str( datetime.datetime.now())) 
        #plt.savefig( '/XF11ID/analysis/Olog_attachments/tmp.png'  )
    plt.savefig( '/XF11ID/analysis/Olog_attachments/%s'%png_name  )    
    plt.show()
class mount_exception(Exception):
    pass



def draw_vert_capillary(x_pos,y_pos,mount_dims):
    ax = plt.gca()
    holder = mpatches.Rectangle([x_pos-1,y_pos-10], 2, 21, angle=0.0, color = 'b',zorder=8)
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])
    

def draw_inair_flat_cell(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([y_pos-13,x_pos-7], 26, 14, angle=0.0, color = 'gray',zorder=3)
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 5, color = 'y',zorder=4)
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 15, color = 'lightgray',zorder=1)
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 10, color = 'white',zorder=2)
    ax = plt.gca()
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_inair_flat_cell(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([y_pos-13,x_pos-7], 26, 14, angle=0.0, color = 'gray',zorder=3)
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 5, color = 'y',zorder=4)
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 15, color = 'lightgray',zorder=1)
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 10, color = 'white',zorder=2)
    ax = plt.gca()
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_wafer_Tcell(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([y_pos-5,x_pos-2], 10, 2, angle=0.0, color = 'red')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Rectangle([y_pos-13,x_pos-5], 30, 3, angle=0.0, color = 'lightgray')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Rectangle([y_pos-16,x_pos-10], 3, 30, angle=0.0, color = 'lightgray')
    ax = plt.gca()
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_horz_capillary(x_pos,y_pos,mount_dims):
    ax = plt.gca()
    holder = mpatches.Rectangle([x_pos-13,y_pos-1], 25, 2, angle=0.0, color = 'b')
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_flat_Tcell(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([y_pos-13,x_pos-7], 26, 14, angle=0.0, color = 'gray')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 5, color = 'y')
    ax.add_artist(holder)
    holder = mpatches.Rectangle([y_pos-13,x_pos-10], 30, 3, angle=0.0, color = 'lightgray')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Rectangle([y_pos-16,x_pos-15], 3, 30, angle=0.0, color = 'lightgray')
    ax = plt.gca()
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_cap_holder(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([x_pos-7,y_pos-13], 14, 35, angle=0.0, color = 'gray',zorder=1)
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Rectangle([x_pos-1,y_pos-10], 2, 40, angle=0.0, color = 'b',zorder=3)
    ax.add_artist(holder)
    holder = mpatches.Rectangle([x_pos-3,y_pos-10], 6, 20, angle=0.0, color = 'white',zorder=2)
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_flatcell(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([x_pos-7,y_pos-13], 14, 26, angle=0.0, color = 'gray')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 5, color = 'y')
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_waf_holder(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([x_pos-7,y_pos-13], 14, 13, angle=0.0, color = 'gray')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Rectangle([x_pos-5,y_pos], 10, 1, angle=0.0, color = 'r')
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_empty_flat_slot(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([x_pos-7,y_pos-13], 14, 35, angle=0.0,edgecolor = 'k',linewidth = 2, linestyle='--', facecolor = 'none',zorder=2)
    ax = plt.gca()
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

def draw_alignment_flat(x_pos,y_pos,mount_dims):
    holder = mpatches.Rectangle([x_pos-7,y_pos-13], 14, 26, angle=0.0, color = 'gray')
    ax = plt.gca()
    ax.add_artist(holder)
    holder = mpatches.Circle([x_pos,y_pos], 2, color = 'r')
    ax.add_artist(holder)
    ax.set_xlim(-mount_dims[0],mount_dims[0])
    ax.set_ylim(-mount_dims[1],mount_dims[1])

# wrapper for database search:

def search_sample_database(filters,show_points=False):
    """
    filters: search string accoring to pymongo database syntax
    show_points=True/False: show/don't show sample[info][points]
    by LW Sept 2018
    """
    x=samples_2.find(filters)
    for docs in x:
        pass
    if x.retrieved >= 1:
        print('Found '+str(x.retrieved)+' database entries for filter: '+str(filters)+'\n')
        x=samples_2.find(filters)
        for docs in x:
            if show_points == False:
                a=docs['info'].pop('points')
            print(docs)
            print('')
    else:
        print('No database entries found for filter: '+str(filters))



# Functions for sample grid creation, update, etc.

def add_new_spot_method(obid,interactive=True,method='from_center'):
    if interactive == True:
        user_input = input('Add new_spot_method for sample database key '+str(obid)+'. Options: "from_center","consecutive","random","exit" : ')
        if user_input == 'exit':
            print(' new_spot_method not updated ')
        else: 
            update_new_spot_method(obid,method=user_input)
    else:
        update_new_spot_method(obid,method=method)

def add_sampling_grid(obid,interactive=True,holder=['flat_cell',1.5,1],x_step=.5,y_step=.5):
    if interactive ==True:
        temp_dict = samples_2.find_one({'_id':  obid})
        if not temp_dict['info']['points'] and temp_dict['sample']['holder'] != 'none': # current list of points is empty
            user_input = input("Current list of sampling points is empty. \nCreate sampling points? yes / no: ")
        else: user_input = input("Sampling grid exists, but might contain insufficient number of points. \nCreate new sampling points? yes / no: ")
        if user_input == 'yes':
            del(user_input)
            lup=0
            while lup ==0:
                #%clear
                [x_point,y_point,dose,fresh_spots]=interactive_sample_grid(obid)
                plt.show()
                user_input=input('Update database with this sampling grid? yes/no/exit: ')
                if user_input == 'yes':
                    del(user_input)
                    update_sample_database_with_new_sampling_grid(obid,x_point,y_point,dose,fresh_spots)
                    lup=1
                elif user_input == 'no':
                    del(user_input)
                    clear_output()
                    pass
                elif user_input == 'exit':
                    del(user_input)
                    print('Exiting without adding sampling grid.')
                    lup=1
        elif user_input =='no':
            print('No sampling points added at this time...!')
            del(user_input)
    else:
        [x_point,y_point,dose,fresh_spots]=define_sampling_grid(holder,x_step=x_step,y_step=y_step)
        update_sample_database_with_new_sampling_grid(obid,x_point,y_point,dose,fresh_spots)

def sample_to_database(sample_dictionary):
    """
    function to insert sample information defined in dictionary to chx sample database
    checks whether entry with indentical sample description and owner exists
    raises an error if sample_dictionary is entirely a duplicate of a database entry
    by LW 08/31/2018
    """
    #check for mandatory fields
    if sam['sample']['sample name'] =='' or sam['info']['owner'] == '':
        print('error: sample name and owner are mandatory fields and cannot be empty!') ### in function, this needs to become a raise exception
        raise chx_database_exception('Error: mandatory fields missing in sample dictionary.')
    #temp_dict = samples_2.find_one({'_id':  obid})

    # check if entry with identical sample description and owner exists:
    x=samples_2.find({ "$and": [{'sample':sam['sample']},{'info.owner':sam['info']['owner']}]})
    for docs in x:
        pass
    if x.retrieved >=1: # = found entries with same sample description and owner
        print('Found already '+str(x.retrieved)+' database entrie(s) with identical sample description and owner:\n')
        x=samples_2.find({ "$and": [{'sample':sam['sample']},{'info.owner':sam['info']['owner']}]})
        for docs in x:
            print(docs),print('\n')
        print('Do you want to continue to add the current sample information to the database?')
        user_input = input("yes / no: ")
        if user_input=='yes':
            del(user_input)
            x=samples_2.insert_one(sam)
            if x.acknowledged == True:
                print('\nInformation successfully added to database!')
                print('New database key added: '+str(x.inserted_id))
                print('New database entry:\n')
                print(samples_2.find_one({'_id':x.inserted_id}))
                print(x.inserted_id)
                return x.inserted_id  
        elif user_input == 'no':
            print('-> information not entered into database')
            del(user_input)
    elif x.retrieved <1: # = found NO entries with same sample description and owner -> update without asking any more questions
        x=samples_2.insert_one(sam)
        if x.acknowledged == True:
            print('\nInformation successfully added to database!')
            print('New database key added: '+str(x.inserted_id))
            print('New database entry:\n')
            print(samples_2.find_one({'_id':x.inserted_id}))
            print(x.inserted_id)
            return x.inserted_id
            
        
class chx_database_exception(Exception):
    pass

def update_label_method(obid,interactive=True,label='none'):
    """
    method to update sample database key 'label'
    obid: ObjectId for existing document in database
    interactive = True: getting current label from database, asking user whether to update it or not; if yes, asks for new label, input 'label' is ignored in this mode
    interactive = False: updating label with provided string 'label'
    by LW 08/31/2018
    """
    old_value=samples_2.find_one({'_id':  obid})['sample']['label']
    if interactive:
        input1 = input('Current label for sample id '+str(obid)+': '+old_value+'. Change label? yes/no: ')
        if input1 == 'yes':
            label = input('Type new label (keep it short!): ')
            samples_2.update_one({'_id':  obid},{'$set':{'sample.label' : label}})
            print('"sample label" for database entry with key '+str(obid)+' updated from '+old_value+' to '+label)
        else: 
            'Sample label not updated...'
    else:
        samples_2.update_one({'_id':  obid},{'$set':{'sample.label' : label}})
        print('"sample label" for database entry with key '+str(obid)+' updated from '+old_value+' to '+label)

def update_new_spot_method(obid,method):
    """
    method to update sample database key 'new_spot_method'
    obid: ObjectId for existing document in database
    method [string]: 'none', 'consecutive', 'from_center', 'random'
    by LW 08/31/2018
    """
    old_value=samples_2.find_one({'_id':  obid})['info']['new_spot_method']
    samples_2.update_one({'_id':  obid},{'$set':{'info.new_spot_method' : method}})
    print('"new_spot_method" for database entry with key '+str(obid)+' updated from '+old_value+' to '+method)

def interactive_sample_grid(obid):
    """
    interactive method to define sample points for sample in database with known holder
    by LW 08/31/2018
    """
    t_dict=samples_2.find_one({'_id':  obid})
    holder = t_dict['sample']['holder']
    print('holder information in database: '+str(holder))
    steps=[]
    x_step=float(input('x_step size [mm]: '))
    y_step=float(input('y_step size [mm]: '))
    print('new sampling points for sample with ObjectID: \n'+str(obid)+' (sample name: '+str(t_dict['sample']['sample name']))
    [x_point,y_point,dose,frash_spots]=define_sampling_grid(t_dict['sample']['holder'],x_step=x_step,y_step=y_step)
    return x_point,y_point,dose,frash_spots

# update 'points' in sample database:
def update_sample_database_with_new_sampling_grid(obid,x_point,y_point,dose,fresh_spots,verbose=False):
    points=[x_point.tolist(),y_point.tolist(),dose.tolist(),fresh_spots]
    samples_2.update_one({'_id':  obid},{'$set':{'info.points' : points}})
    if verbose:
        print('sample information has been updated with new points for data acquisition:\n')
        print(samples_2.find_one({'_id':obid}))



def define_sampling_grid(holder,x_step=.1,y_step=.1):
    """
    input parameters:
    holder=['flat_cell',1.5]   # flat cell, radius of aperture [mm]
    holder=['capillary',1.0,[-4,-1],'vertical']  # capillary, diameter [mm], useful height/width (depending on orientation) [min,max] (relative to center), widht of the capillary used is 1/4 x diameter, orientation='horizontal'/'vertical'
    holder=['wafer',.5,[2,4]] # wafer, thickness, useful width [min,max] (relative to center)
    returns: x_point, y_point -> defining points available for measurements, dose: same length as x_point and y_point: array to track dose received in every point
    by LW 08/28/2018
    """
    # number of rows and columns needed:
    if holder[0] == 'flat_cell':
        grid_height = 2*holder[1]
        grid_width = 2*holder[1]
    elif holder[0] == 'capillary':
        grid_width = holder[1]*.25
        grid_height = np.abs(holder[2][1]*.9-holder[2][0]*.9)
        grid_offset = np.mean(holder[2])
    elif holder[0] =='wafer':
        grid_width = np.abs(holder[2][1]*.9-holder[2][0]*.95)
        grid_offset = np.mean(holder[2])
        grid_yoffset = holder[1]
        grid_height = 1

    rown=np.ceil(grid_width/x_step)
    coln=np.ceil(grid_height/y_step)+1

    # create regular grid (including offset from row to row)
    x=np.arange(-grid_width/2,grid_width/2,x_step)
    x_point=[]
    y_point=[]
    if holder[0]!='wafer':
        for i in range(0,int(coln)):  
            x_point=x_point+list(x+(-1)**i*x_step/4)
            y_point=y_point+list(np.ones(np.shape(x))*(-grid_height/2)+i*y_step)
    elif holder[0]=='wafer':
        x_point=x+grid_offset
        y_point=np.zeros(np.shape(x_point))+grid_yoffset
    if holder[0] == 'flat_cell':
        in_aperture = np.array(x_point)**2+np.array(y_point)**2 <= (holder[1]*.9)**2
    elif holder[0] == 'capillary':
        y_point=y_point+grid_offset
        in_aperture = np.abs(np.array(x_point)) <= grid_width/2
    elif holder[0] == 'wafer':
        in_aperture=np.ones(np.shape(x_point), dtype=bool)
    # can have capillary on vertical or horizontal (sample chamber) orientation:
    if holder[0] == 'capillary' and holder[3] == 'horizontal':
        tx=x_point;ty= y_point
        x_point=ty;y_point=tx

    print('sample holder: '+str(holder[0]))
    print('step size x: '+str(x_step)+'   step size y: '+str(y_step)+'   number of points available: '+str(len(np.array(x_point)[in_aperture])))
    plt.subplot()
    plt.plot(x_point,y_point,'k+')
    plt.plot(np.array(x_point)[in_aperture],np.array(y_point)[in_aperture],'ro')
    if holder[0] == 'flat_cell':
        an = np.linspace(0, 2*np.pi, 100)
        plt.plot(holder[1]*np.cos(an), holder[1]*np.sin(an),'k-',linewidth=3)
        plt.axis('equal')
    elif holder[0] == 'capillary' and holder[3] == 'vertical':
        plt.plot([-grid_width/2,-grid_width/2,grid_width/2,grid_width/2,-grid_width/2],[holder[2][0],holder[2][1],holder[2][1],holder[2][0],holder[2][0]],'k-',linewidth=3)
    elif holder[0] == 'capillary' and holder[3] == 'horizontal':
        plt.plot([holder[2][0],holder[2][1],holder[2][1],holder[2][0],holder[2][0]],[-grid_width/2,-grid_width/2,grid_width/2,grid_width/2,-grid_width/2],'k-',linewidth=3)
    elif holder[0] == 'wafer':
        plt.plot([-grid_width/2,-grid_width/2,grid_width/2,grid_width/2,-grid_width/2]+grid_offset,[grid_yoffset-.5,grid_yoffset+.5,grid_yoffset+.5,grid_yoffset-.5,grid_yoffset-.5],'k-',linewidth=3)
        #plt.axis('equal')
    plt.grid()
    plt.xlabel('x-position [mm]');plt.ylabel('y-position [mm]')

    x_point=np.array(x_point)[in_aperture]
    y_point=np.array(y_point)[in_aperture]
    dose=np.zeros(np.shape(x_point))
    fresh_spots=len(dose==0)
    return x_point,y_point,dose,fresh_spots

def visualize_sampling_grid(holder,x_point,y_point,dose,com=''):
    """
    function to visualize sampling grid -and dose individual points received so far- created by define_sampling_grid
    input parameters:
    - holder -> same as for define_sampling_grid
    - x_point | y_point | dose: created by define sampling grid
    com [string]: comment to be shown as part of plot title -> 'Dose map for: '+com+' [sec of full beam]' 
    by LW 08/28/2018
    """
    x_plot = x_point #np.array(x_point)
    y_plot = y_point #np.array(y_point)
    
    if holder[0] == 'flat_cell':
        grid_height = 2*holder[1]
        grid_width = 2*holder[1]
    elif holder[0] == 'capillary':
        grid_width = holder[1]*.25
        grid_height = np.abs(holder[2][1]*.9-holder[2][0]*.9)
        grid_offset = np.mean(holder[2])
    elif holder[0] =='wafer':
        grid_width = np.abs(holder[2][1]*.9-holder[2][0]*.95)
        grid_offset = np.mean(holder[2])
        grid_yoffset = holder[1]
        grid_height = 1
    
    plt.subplot(1,1,1)
    plt.scatter(x_plot[dose!=0],y_plot[dose!=0],c=dose[dose!=0],cmap='plasma')
    plt.colorbar()
    plt.scatter(x_plot[dose==0],y_plot[dose==0],c='g',marker='o',label='dose=0s')
    plt.legend(loc=3)
    # countour of usefull sample area:
    if holder[0] == 'flat_cell':
        an = np.linspace(0, 2*np.pi, 100)
        plt.plot(holder[1]*np.cos(an), holder[1]*np.sin(an),'k-',linewidth=3)
        plt.axis('equal')
    elif holder[0] == 'capillary' and holder[3] == 'vertical':
        plt.plot([-grid_width/2,-grid_width/2,grid_width/2,grid_width/2,-grid_width/2],[holder[2][0],holder[2][1],holder[2][1],holder[2][0],holder[2][0]],'k-',linewidth=3)
    elif holder[0] == 'capillary' and holder[3] == 'horizontal':
        plt.plot([holder[2][0],holder[2][1],holder[2][1],holder[2][0],holder[2][0]],[-grid_width/2,-grid_width/2,grid_width/2,grid_width/2,-grid_width/2],'k-',linewidth=3)
    elif holder[0] == 'wafer':
        plt.plot([-grid_width/2,-grid_width/2,grid_width/2,grid_width/2,-grid_width/2]+grid_offset,[grid_yoffset-.5,grid_yoffset+.5,grid_yoffset+.5,grid_yoffset-.5,grid_yoffset-.5],'k-',linewidth=3)
        plt.axis('equal')
    plt.title('Dose map for: '+com+' [sec of full beam]')
    plt.xlabel('x-position [mm]');plt.ylabel('y-position [mm]')
    plt.grid()

def next_grid_point(x_point,y_point,dose,mode='consecutive'):
    """
    
    by LW 08/28/2018
    """
    # points with zero dose available:
    x_available=x_point[dose==0]
    y_available=y_point[dose==0]
    
    # center of grid:
    x_cen=np.mean(x_point)
    y_cen=np.mean(y_point)
    
    # next point to be used:
    if mode == 'from_center':
        distance=(x_available-x_cen)**2+(y_available-y_cen)**2
        next_ind=np.argmin(distance)
    elif mode == 'consecutive':
        next_ind = 0
    elif mode == 'random':
        next_ind=np.random.randint(0,len(x_available)-1)
    
    next_x_point=x_available[next_ind]
    next_y_point=y_available[next_ind]
    # need to find index of that point in the whole grid (to update dose)
    a=x_point == x_available[next_ind]
    b=y_point == y_available[next_ind]
    dose_ind= [i for i, x in enumerate(a*b) if x]
    return [dose_ind[0],next_x_point,next_y_point]

def get_n_fresh_spots(obid):
    t_dict=samples_2.find_one({'_id':obid})
    try: 
        n_points=t_dict['info']['points'][3]
        return n_points
    except:
        return False






















