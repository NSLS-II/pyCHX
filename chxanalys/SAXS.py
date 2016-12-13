"""
Sep 10 Developed by Y.G.@CHX 
yuzhang@bnl.gov
This module is for the static SAXS analysis, such as fit form factor
"""

#import numpy as np
#from lmfit import  Model
#from lmfit import minimize, Parameters, Parameter, report_fit
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from chxanalys.chx_libs import *
from chxanalys.chx_generic_functions import show_img

def mono_sphere_form_factor_intensity( x, radius, delta_rho=100):
    '''
    Input:
        x/q: in A-1, array or a value
        radius/R: in A

        delta_rho: Scattering Length Density(SLD) difference between solvent and the scatter, A-2
    Output:
        The form factor intensity of the mono dispersed scatter    
    '''
    q=x
    R=radius
    qR= q * R        
    volume = (4.0/3.0)*np.pi*(R**3)
    prefactor = 36*np.pi*( ( delta_rho * volume)**2 )/(4*np.pi)
    P = ( np.sin(qR) - qR*np.cos(qR) )**2/( qR**6 )
    P*=prefactor
    P=P.real
    return P


def gaussion( x, u, sigma):
    return 1/( sigma*np.sqrt(2*np.pi) )* np.exp( - ((x-u)**2)/(2*( sigma**2 ) ) )


def distribution_gaussian( radius=1.0, sigma=0.1, num_points=20, spread=3):
    '''
    radius: the central radius
    sigma: sqrt root of variance in percent
    '''
    
    if 1 - spread* sigma<=0:
        spread= (1 - sigma)/sigma -1
        
    x, rs= np.linspace( radius - radius*spread* sigma, radius+radius*spread*sigma, num_points,retstep=True)
    
    return x, rs, gaussion( x, radius, radius*sigma)*rs
    
    
def poly_sphere_form_factor_intensity( x, radius, sigma=0.1, delta_rho=100):
    '''
    Input:
        x/q: in A-1, array or a value
        radius/R: in A
        sigma:sqrt root of variance in percent
        delta_rho: Scattering Length Density(SLD) difference between solvent and the scatter, A-2
    Output:
        The form factor intensity of the polydispersed scatter    
    '''    
    q=x
    R= radius
    if not hasattr(q, '__iter__'):
        q=np.array( [q] )
    v = np.zeros( (len(q)) )
    if sigma==0:
        v= mono_sphere_form_factor_intensity( q, R, delta_rho)
    else:
        r, rs, wt = distribution_gaussian( radius=R, sigma=sigma, num_points=20, spread=5)
        for i, Ri in enumerate(r):
            v += mono_sphere_form_factor_intensity( q, Ri, delta_rho)*wt[i]*rs
    return v #* delta_rho


def poly_sphere_form_factor_intensity_q2( x, radius, sigma=0.1, delta_rho=1):#, scale=1, baseline=0):
    '''
    Input:
        x/q: in A-1, array or a value
        radius/R: in A
        sigma:sqrt root of variance in percent
        delta_rho: Scattering Length Density(SLD) difference between solvent and the scatter, A-2
    Output:
        The form factor intensity of the polydispersed scatter    
    ''' 
    
    return poly_sphere_form_factor_intensity( x, radius, sigma, delta_rho)*x**2 #* scale + baseline
    

def find_index( x,x0,tolerance= None):
    #find the position of P in a list (plist) with tolerance

    N=len(x)
    i=0
    position=None
    if tolerance==None:
        tolerance = (x[1]-x[0])/2.
    for item in x:
        if abs(item-x0)<=tolerance:
            position=i
            #print 'Found Index!!!'
            break
        i+=1
    return position


def get_form_factor_fit( q, iq, guess_values, fit_range=None, fit_variables = None,function='poly_sphere', 
                         *argv,**kwargs): 
    '''
    Fit form factor for GUI
    
    The support fitting functions include  
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ---------- 
    q: q vector
    iq: form factor
    
    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent 
        
    function: 
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model
                    
    Returns
    -------        
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    '''
    
    if function=='poly_sphere':        
        mod = Model(poly_sphere_form_factor_intensity_q2 )
    elif function=='mono_sphere':  
        mod = Model( mono_sphere_form_factor_intensity )        
    else:
        print ("The %s is not supported.The supported functions include poly_sphere and mono_sphere"%function)
        
    if fit_range is not None:
        x1,x2= fit_range
        q1=find_index( q,x1,tolerance= None)
        q2=find_index( q,x2,tolerance= None)            
    else:
        q1=0
        q2=len(q)

    q_=q[q1:q2]
    iq_ = iq[q1:q2]        
        
    _r= guess_values[ 'radius']
    _sigma = guess_values['sigma']
    _delta_rho= guess_values['delta_rho']
    #_scale = guess_values['scale']
    #_baseline = guess_values['baseline']
    
    mod.set_param_hint( 'radius',   min= _r/10, max=_r*10  )
    mod.set_param_hint( 'sigma',   min= _sigma/10, max=_sigma*10 ) 
    #mod.set_param_hint( 'scale',   min= _scale/1E3, max= _scale*1E3  )   
    #mod.set_param_hint( 'baseline',   min= 0  )   
    #mod.set_param_hint( 'delta_rho',   min= 0  )  
    mod.set_param_hint( 'delta_rho',   min= _delta_rho/1E6, max= _delta_rho*1E6  )  
    pars  = mod.make_params( radius= _r, sigma=_sigma,delta_rho=_delta_rho)# scale= _scale, baseline =_baseline  )  
    
    if fit_variables is not None:
        for var in  list( fit_variables.keys()):
            pars[var].vary = fit_variables[var]            
    #pars['delta_rho'].vary =False       
    fit_power = 2    
    result = mod.fit( iq_* q_**fit_power, pars, x = q_ )    
    if function=='poly_sphere':        
        sigma = result.best_values['sigma']
    elif function=='mono_sphere':  
        sigma=0        
    r = result.best_values['radius']  
    #scale =  result.best_values['scale']
    #baseline = result.best_values['baseline']
    delta_rho= result.best_values['delta_rho']    
    return result, q_

def plot_form_factor_with_fit(q, iq, q_, result,  fit_power=2,  res_pargs=None, return_fig=False,
                         *argv,**kwargs):
    
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path']     
    else:    
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'
        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
    
    fig = Figure()
    ax = fig.add_subplot(111)        
    title_qr = 'form_factor_fit'        
    plt.title('uid= %s:--->'%uid + title_qr,fontsize=20, y =1.02)
    
    r = result.best_values['radius'] 
    delta_rho= result.best_values['delta_rho']
    sigma = result.best_values['sigma'] 
        
    ax.semilogy( q, iq, 'ro', label='Form Factor')        
    ax.semilogy( q_, result.best_fit/q_**fit_power, '-b', lw=3, label='Fit')            
    
    txts = r'radius' + r' = %.2f '%( r/10.) +   r'$ nm$' 
    ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)
    txts = r'sigma' + r' = %.3f'%( sigma)  
    #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    ax.text(x =0.02, y=.25, s=txts, fontsize=14, transform=ax.transAxes)  
    #txts = r'delta_rho' + r' = %.3e'%( delta_rho)  
    #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    #ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)  
    ax.legend( loc = 'best' )

    if 'ylim' in kwargs:
        ax.set_ylim( kwargs['ylim'])
    elif 'vlim' in kwargs:
        vmin, vmax =kwargs['vlim']
        ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
    else:
        pass
    if 'xlim' in kwargs:
        ax.set_xlim( kwargs['xlim'])

    fp = path + 'uid=%s--form_factor--fit-'%(uid )  + '.png'
    plt.savefig( fp, dpi=fig.dpi)        
    fig.tight_layout()  
    plt.show()

    if return_fig:
        return fig


    
def fit_form_factor( q, iq,  guess_values, fit_range=None, fit_variables = None, res_pargs=None,function='poly_sphere', 
                         *argv,**kwargs):    
     
    '''
    Fit form factor
    
    The support fitting functions include  
                poly_sphere (poly_sphere_form_factor_intensity),
                mono_sphere (mono_sphere_form_factor_intensity)
    Parameters
    ---------- 
    q: q vector
    iq: form factor
    res_pargs: a dict, contains keys, such path, uid...
    
    guess_values:a dict, contains keys
        radius: the initial guess of spherecentral radius
        sigma: the initial guess of sqrt root of variance in percent 
        
    function: 
        mono_sphere (mono_sphere_form_factor_intensity): fit by mono dispersed sphere model
        poly_sphere (poly_sphere_form_factor_intensity): fit by poly dispersed sphere model
                    
    Returns
    -------        
    fit resutls:
         radius
         sigma
    an example:
        result = fit_form_factor( q, iq,  res_pargs=None,function='poly_sphere'
    '''

    
    if res_pargs is not None:
        uid = res_pargs['uid'] 
        path = res_pargs['path']     
    else:    
        if 'uid' in kwargs.keys():
            uid = kwargs['uid'] 
        else:
            uid = 'uid'
        if 'path' in kwargs.keys():
            path = kwargs['path'] 
        else:
            path = ''
 
    
    if function=='poly_sphere':        
        mod = Model(poly_sphere_form_factor_intensity_q2 )
    elif function=='mono_sphere':  
        mod = Model( mono_sphere_form_factor_intensity )        
    else:
        print ("The %s is not supported.The supported functions include poly_sphere and mono_sphere"%function)
        
    if fit_range is not None:
        x1,x2= fit_range
        q1=find_index( q,x1,tolerance= None)
        q2=find_index( q,x2,tolerance= None)            
    else:
        q1=0
        q2=len(q)

    q_=q[q1:q2]
    iq_ = iq[q1:q2]        
        
    _r= guess_values[ 'radius']
    _sigma = guess_values['sigma']
    _delta_rho= guess_values['delta_rho']
    #_scale = guess_values['scale']
    #_baseline = guess_values['baseline']
    
    mod.set_param_hint( 'radius',   min= _r/10, max=_r*10  )
    mod.set_param_hint( 'sigma',   min= _sigma/10, max=_sigma*10 ) 
    #mod.set_param_hint( 'scale',   min= _scale/1E3, max= _scale*1E3  )   
    #mod.set_param_hint( 'baseline',   min= 0  )   
    #mod.set_param_hint( 'delta_rho',   min= 0  )  
    mod.set_param_hint( 'delta_rho',   min= _delta_rho/1E6, max= _delta_rho*1E6  )  
    pars  = mod.make_params( radius= _r, sigma=_sigma,delta_rho=_delta_rho)# scale= _scale, baseline =_baseline  )  
    
    if fit_variables is not None:
        for var in  list( fit_variables.keys()):
            pars[var].vary = fit_variables[var]            
    #pars['delta_rho'].vary =False
    
    fig = plt.figure(figsize=(8, 6))
    title_qr = 'form_factor_fit'        
    plt.title('uid= %s:--->'%uid + title_qr,fontsize=20, y =1.02)
       
    fit_power = 2
    
    result = mod.fit( iq_* q_**fit_power, pars, x = q_ )
    
    if function=='poly_sphere':        
        sigma = result.best_values['sigma']
    elif function=='mono_sphere':  
        sigma=0
        
    r = result.best_values['radius']  
    #scale =  result.best_values['scale']
    #baseline = result.best_values['baseline']
    delta_rho= result.best_values['delta_rho']
    
    #report_fit( result )
    
    ax = fig.add_subplot(1,1,1 )   
    
    ax.semilogy( q, iq, 'ro', label='Form Factor')        
    ax.semilogy( q_, result.best_fit/q_**fit_power, '-b', lw=3, label='Fit')            
    
    txts = r'radius' + r' = %.2f '%( r/10.) +   r'$ nm$' 
    ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)
    txts = r'sigma' + r' = %.3f'%( sigma)  
    #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    ax.text(x =0.02, y=.25, s=txts, fontsize=14, transform=ax.transAxes)  
    #txts = r'delta_rho' + r' = %.3e'%( delta_rho)  
    #txts = r'$\beta$' + r'$ = %.3f$'%(beta[i]) +  r'$ s^{-1}$'
    #ax.text(x =0.02, y=.35, s=txts, fontsize=14, transform=ax.transAxes)  
    ax.legend( loc = 'best' )

    if 'ylim' in kwargs:
        ax.set_ylim( kwargs['ylim'])
    elif 'vlim' in kwargs:
        vmin, vmax =kwargs['vlim']
        ax.set_ylim([min(y)*vmin, max(y[1:])*vmax ])
    else:
        pass
    if 'xlim' in kwargs:
        ax.set_xlim( kwargs['xlim'])

    fp = path + 'uid=%s--form_factor--fit-'%(uid )  + '.png'
    fig.savefig( fp, dpi=fig.dpi)        
    fig.tight_layout()  
    plt.show()

    result = dict( radius =r, sigma = sigma, delta_rho = delta_rho  )

    return result
                

    
    
def show_saxs_qmap( img, pargs, width=200,vmin=.1, vmax=300, logs=True,image_name='', 
                   save=False, show_pixel=False):
    '''
    Show a SAXS q-map by giving 
    Parameter:
        image: the frame
        setup pargs, a dictionary, including
            dpix    #in mm, eiger 4m is 0.075 mm
            lambda_     # wavelegth of the X-rays in Angstroms
            Ldet     # detector to sample distance (mm)
            path  where to save data
            center: beam center in pixel
        width: the showed area centered at center
    Return:
        None 
    '''
    
    Ldet = pargs['Ldet']
    dpix = pargs['dpix']
    lambda_ =  pargs['lambda_']    
    center = pargs['center']
    path= pargs['path']
    
    
    w= width
  
    if not show_pixel:
        two_theta = utils.radius_to_twotheta(Ldet, [w * dpix])
        qext  = utils.twotheta_to_q(two_theta, lambda_)[0] 
        show_img( 1e-15+ img[center[0]-w:center[0]+w, center[1]-w:center[1]+w], 
        xlabel=r"$q_x$" +  '('+r'$\AA^{-1}$'+')', 
         ylabel= r"$q_y$" +  '('+r'$\AA^{-1}$'+')', extent=[-qext,qext,-qext,qext],
            vmin=vmin, vmax=vmax, logs= logs, image_name= image_name, save= save, path=path) 
    else:        
        #qext = w
        show_img( 1e-15+ img[center[0]-w:center[0]+w, center[1]-w:center[1]+w], 
        xlabel= 'pixel',  ylabel= 'pixel', 
            vmin=vmin, vmax=vmax, logs= logs, image_name= image_name, save= save, path=path)     
    


def exm_plot():
    fig, ax = plt.subplots()

    ax.semilogy( q, iq, 'ro',label='data') 
    ax.semilogy( q, ff, '-b',label='fit') 
    ax.set_xlim( [0.0001, .01] )
    ax.set_ylim( [1E-2,1E4] )
    ax.legend(loc='best')
    #plot1D( iq, q, logy=True, xlim=[0.0001, .01], ylim=[1E-3,1E4], ax=ax, legend='data')
    #plot1D( ff, q, logy=True, xlim=[0.0001, .01], ax=ax, legend='cal')

    #%run /XF11ID/analysis/Analysis_Pipelines/Develop/chxanalys/chxanalys/XPCS_SAXS.py
    #%run /XF11ID/analysis/Analysis_Pipelines/Develop/chxanalys/chxanalys/chx_generic_functions.py
    #%run /XF11ID/analysis/Analysis_Pipelines/Develop/chxanalys/chxanalys/SAXS.py
