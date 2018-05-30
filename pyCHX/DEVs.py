#simple brute force multitau
#from pyCHX.chx_generic_functions import average_array_withNan
import numpy as np
from tqdm import tqdm

def multitau(Ipix,bind,lvl=12,nobuf=8):
    '''
   tt,g2=multitau(Ipix,bind,lvl=12,nobuf=8)
      Ipix is matrix of no-frames by no-pixels
      bind is bin indicator, one per pixel. All pixels
      in the same bin have the same value of bind (bin indicator).
      This number of images save at each level is nobf(8) and the
      number of level is lvl(12).
    returns:
    tt is the offsets of each g2 point. 
		tt*timeperstep is time of points
	g2 is max(bind)+1,time steps.
	   plot(tt[1:],g2[1:,i]) will plot each g2.
    '''
    #if num_lev is None:
    #    num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    #print(nobuf,nolvl)
    nobins=bind.max()+1
    nobufov2=nobuf//2
    #t0=time.time()
    noperbin=np.bincount(bind)
    G2=np.zeros((nobufov2*(1+lvl),nobins))
    tt=np.zeros(nobufov2*(1+lvl))
    dII=Ipix.copy()
    t=np.bincount(bind,np.mean(dII,axis=0))**2/noperbin
    G2[0,:]=np.bincount(bind,np.mean(dII*dII,axis=0))/t
    for j in np.arange(1,nobuf):
        tt[j]=j
        print(j,tt[j])
        #t=noperbin*np.bincount(bind,np.mean(dII,axis=0))**2
        t=np.bincount(bind,np.mean(dII[j:,:],axis=0))*np.bincount(bind,np.mean(dII[:-j,:],axis=0))/noperbin
        G2[j,:] = np.bincount(bind,np.mean(dII[j:,:]*dII[:-j,:],axis=0))/t
    for l in tqdm( np.arange(1,lvl), desc= 'Calcuate g2...'  ):
        nn=dII.shape[0]//2*2 #make it even
        dII=(dII[0:nn:2,:]+dII[1:nn:2,:])/2.0 #sum in pairs
        nn=nn//2
        if(nn<nobuf): break
        for j in np.arange(nobufov2,min(nobuf,nn)):
            ind=nobufov2+nobufov2*l+(j-nobufov2)
            tt[ind]=2**l*j
            t=np.bincount(bind,np.mean(dII[j:,:],axis=0))*np.bincount(bind,np.mean(dII[:-j,:],axis=0))/noperbin
            G2[ind,:] = np.bincount(bind,np.mean(dII[j:,:]*dII[:-j,:],axis=0))/t
    #print(ind)
    #print(time.time()-t0)
    return(tt[:ind+1],G2[:ind+1,:])



def average_array_withNan( array,  axis=0, mask=None):
    '''YG. Jan 23, 2018
       Average array invovling np.nan along axis       
        
       Input:
           array: ND array, actually should be oneD or twoD at this stage..TODOLIST for ND
           axis: the average axis
           mask: bool, same shape as array, if None, will mask all the nan values 
       Output:
           avg: averaged array along axis
    '''
    shape = array.shape
    if mask is None:
        mask = np.isnan(array)
        #mask = np.ma.masked_invalid(array).mask 
    array_ = np.ma.masked_array(array, mask=mask) 
    try:
        sums = np.array( np.ma.sum( array_[:,:], axis= axis ) )
    except:
        sums = np.array( np.ma.sum( array_[:], axis= axis ) )
        
    cts = np.sum(~mask,axis=axis)
    #print(cts)
    return sums/cts



def autocor_for_pix_time( pix_time_data,  dly_dict,  pixel_norm = None, frame_norm = None, 
                         multi_tau_method = True ):
    '''YG Feb 20, 2018@CHX
    Do correlation for pixel_time type data with tau as defined as dly
    Input:
        pix_time_data: 2D array, shape as [number of frame (time), pixel list (each as one q) ]        
        dly_dict: the taus dict, (use multi step e.g.)
        multi_tau_method: if True, using multi_tau_method to average data for higher level 
        pixel_norm: if not None, should be array with shape as pixel number
        frame_norm: if not None, should be array with shape as frame number
        
    return:
        g2: with shape as [ pixel_list number, dly number ] 
    
    ''' 
    Nt, Np = pix_time_data.shape
    Ntau =  len(np.concatenate( list(dly_dict.values() ) ))     
    G2=np.zeros( [ Ntau, Np ] )    
    Gp=np.zeros( [ Ntau, Np ] ) 
    Gf=np.zeros( [ Ntau, Np ] ) 
    #mask_pix = np.isnan(pix_time_data)
    #for tau_ind, tau in tqdm( enumerate(dly), desc= 'Calcuate g2...'  ): 
    tau_ind = 0 
    #if multi_tau_method:
    pix_time_datac = pix_time_data.copy()
    #else:
    if pixel_norm is not None:
        pix_time_datac /= pixel_norm
    if frame_norm is not None:
        pix_time_datac /= frame_norm        
            
    for tau_lev, tau_key in tqdm( enumerate( list(dly_dict.keys() ) ), desc= 'Calcuate g2...'  ):
        #print(tau_key)
        taus = dly_dict[tau_key]        
        if multi_tau_method:
            if tau_lev >0:
                nobuf = len(dly_dict[1])
                nn=pix_time_datac.shape[0]//2*2 #make it even
                pix_time_datac=(pix_time_datac[0:nn:2,:]+pix_time_datac[1:nn:2,:])/2.0 #sum in pairs
                nn=nn//2
                if(nn<nobuf): break
                #print(nn)
                #if(nn<1): break
            else:
                nn=pix_time_datac.shape[0]
        for tau in taus:
            if multi_tau_method:                   
                IP= pix_time_datac[: nn - tau//2**tau_lev,:]
                IF= pix_time_datac[tau//2**tau_lev: nn,: ] 
                #print( tau_ind, nn ,  tau//2**tau_lev, tau_lev+1,tau, IP.shape) 
            else:
                IP= pix_time_datac[: Nt - tau,:]
                IF= pix_time_datac[tau: Nt,: ] 
                #print( tau_ind,   tau_lev+1,tau, IP.shape) 
                
            #IP_mask = mask_pix[: Nt - tau,:]
            #IF_mask = mask_pix[tau: Nt,: ] 
            #IPF_mask  = IP_mask | IF_mask 
            #IPFm = average_array_withNan(IP*IF, axis = 0, )#mask= IPF_mask )
            #IPm =  average_array_withNan(IP, axis = 0, )#   mask= IP_mask )
            #IFm =  average_array_withNan(IF, axis = 0 , )#  mask= IF_mask )     
            G2[tau_ind] =   average_array_withNan(IP*IF, axis = 0, )#IPFm
            Gp[tau_ind] =   average_array_withNan(IP, axis = 0, )#IPm
            Gf[tau_ind] =   average_array_withNan(IF, axis = 0 , )#IFm  
            tau_ind += 1            
    #for i in range(G2.shape[0]-1, 0, -1):
    #    if np.isnan(G2[i,0]):
    #        gmax = i            
    gmax = tau_ind
    return G2[:gmax,:], Gp[:gmax,:],Gf[:gmax,:]
    
 



def autocor_xytframe(self, n):
    '''Do correlation for one xyt frame--with data name as n '''

    data = read_xyt_frame( n )  #load data
    N = len( data )
    crl = correlate(data,data,'full')[N-1:]
    FN = arange( 1,N+1, dtype=float)[::-1]
    IP = cumsum(data)[::-1]
    IF = cumsum(data[::-1])[::-1]

    return crl/(IP*IF) * FN



###################For Fit

import numpy as np 
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# duplicate my curfit function from yorick, except use sigma and not w
# notice the main feature is an adjust list.

def curfit(x,y,a, sigy=None,function_name=None,adj=None):
    a = np.array(a)
    if(adj is  None): adj=np.arange(len(a), dtype='int32')
    if(function_name is None): function_name=funct
    #print( a, adj, a[adj] )
    #print(x,y,a)
    afit, cv, idt,m,ie = leastsq(_residuals,a[adj],args=(x,y,sigy,a,adj,function_name),full_output=True)
    a[adj]=afit
    realcv=np.identity(afit.size)
    realcv[np.ix_(adj,adj)] = cv
    nresids=idt['fvec']
    chisq=np.sum(nresids**2)/(len(y)-len(adj))
    #print( cv )
    #yfit=y-yfit*sigy
    sigmaa=np.zeros(len(a))
    sigmaa[adj]=np.sqrt(np.diag(cv));
    return (chisq,a,sigmaa,nresids)

#hidden residuals for leastsq
def _residuals(p, x, y, sigy,pall,adj,fun):
    #print(p, pall, adj )
    pall[adj]=p
    #print(pall)
    if sigy is None:
        return (y-fun(x,pall))
    else:
        return (y-fun(x,pall))/sigy

#print out fit result nicely
def fitpr(chisq,a,sigmaa,title=None,lbl=None):
    ''' nicely print out results of a fit '''
    #get fitted results.
    if(lbl==None):
        lbl=[]
        for i in xrange(a.size):
            lbl.append('A%(#)02d'%{'#':i})
    #print resuls of a fit.
    if(title!=None): print(title)
    print('   chisq=%(c).4f'%{'c':chisq})
    for i in range(a.size):
        print('     %(lbl)8s =%(m)10.4f +/- %(s).4f'%{'lbl':lbl[i],'m':a[i],'s':sigmaa[i]})

#easy plot for fit
def fitplot(x,y,sigy,yfit,pl=plt):
    
    pl.plot(x, yfit)
    pl.errorbar(x,y,fmt='o',yerr=sigy)

#define a default function, a straight line.
def funct(x, p):
    return p[0] * x + p[1]

#A 1D Gaussian
def Gaussian( x,  p ):  
    '''
    One-D Gaussian Function
    xo, amplitude, sigma, offset  = p
    
    
    ''' 
    xo, amplitude, sigma, offset  = p
    g = offset + amplitude*1.0/( sigma * np.sqrt(2*np.pi)  ) * np.exp( -1/2. * (x-xo)**2/sigma**2  )
    return g







