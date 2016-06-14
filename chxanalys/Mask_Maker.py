#from .chx_libs import *
from .chx_libs importdb, get_images, get_table, get_events, get_fields, np, plt , LogNorm
from .chx_generic_functions import show_img


#####
#load data by databroker    

    
def get_sid_filenames(header):
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
    events = get_events( header, keys, handler_overrides={key: RawHandler for key in keys})
    key, = keys   
    filenames =  [  str( ev['data'][key][0]) + '_'+ str(ev['data'][key][2]['seq_id']) for ev in events]     
    sid = header['start']['scan_id']
    uid= header['start']['uid']
    
    return sid,uid, filenames   


def load_data( uid , detector = 'eiger4m_single_image'  ):
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

def get_frames_from_dscan(  hdr, detector = 'eiger4m_single_image' ):
    ev = get_events(hdr, [detector]) 
    length = int( hdr['start']['plan_args']['num'] )
    shape = hdr['descriptors'][0]['data_keys'][detector]['shape'][:2]
    imgs = np.zeros( [  length, shape[1],shape[0]]  )
    for i in range(   int( hdr['start']['plan_args']['num'] ) ):
        imgs[i] =  next(ev)['data']['eiger4m_single_image'][0]   
        
    return np.array( imgs )

def load_metadata(hdr, name):
    seq_of_img_stacks = get_images(hdr, name)
    return seq_of_img_stacks[0].md
def load_images(hdr, name):
    seq_of_img_stacks = get_images(hdr, name)  # 1 x 21 x 2167 x 2070
    return np.squeeze(np.asarray(seq_of_img_stacks))


def RemoveHot( img,threshold= 1E7, plot_=True ):
    mask = np.ones_like( np.array( img )    )
    badp = np.where(  np.array(img) >= threshold )
    if len(badp[0])!=0:                
        mask[badp] = 0  
    if plot_:
        show_img( mask )
    return mask


############
###plot data


def get_avg_img( data_series, sampling = 100, plot_ = False ,  *argv,**kwargs):   
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
        ax.set_title('Uid= %s--Masked Averaged Image'%uid)
        fig.colorbar(im)
        plt.show()
    return avg_img




