#from .chx_libs import *
from .chx_libs import db, get_images, get_table, get_events, get_fields, np, plt , LogNorm
from .chx_generic_functions import show_img



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

def test():
    pass

def RemoveHot( img,threshold= 1E7, plot_=True ):
    mask = np.ones_like( np.array( img )    )
    badp = np.where(  np.array(img) >= threshold )
    if len(badp[0])!=0:                
        mask[badp] = 0  
    if plot_:
        show_img( mask )
    return mask






