import numpy as np
from lmfit import Parameters, Model

class VectorField2DFitter:
    ''' Base class for fitting a 2D vector field.
        Must be inherited.
    '''
    def __init__(self, params=None):
        if params is None:
            raise ValueError("Sorry parameters not set, cannot continue")
        self.params = params

        self._res = None

    def __call__(self, x, y, vx, vy, **kwargs):
        ''' The call function will fit the 2D vector field [vx, vy]
            to the fit function specified in the object.
   fig,ax=plt.subplots()
qmag=np.hypot(qxs[w],qys[w])
ax.scatter(qmag,vecmags[w])         Need to specify components and vectors.

            Parameters
            ---------

            x : 2D np.ndarray
                x component

            y : 2D np.ndarray
                y component

            vx : 2D np.ndarray
                the x component of the vector

            vy : 2D np.ndarray
                the y component of the vector

            kwargs : the initial guesses.

            Returns
            -------
            The best fit
        '''
        guesskeys = self.guess()
        params = self.params.copy()

        # make the parameters from the kwargs
        for key in self.params.keys():
            if key in kwargs.keys() and key is not 'XY':
                params[key].value = kwargs[key]
            else:
                # then guess
                params[key].value = guesskeys[key]

        self.mod = Model(self.fitfunc, independent_vars=['x', 'y'],
                         param_names = self.params.keys())
        # assumes first var is dependent var, and save last params
        V = np.array([vx, vy])
        self._res = self.mod.fit(V, x=x, y=y, params=params)
        self._x = x
        self._y = y
        self._params = params

        return self._res.best_values

    def last_result(self):
        ''' Return fitted result of the last fit.'''
        if self._res is None:
            return ValueError("Please run fit first")
        return self._res.best_fit

    def last_values(self):
        ''' Return fitted values of the last fit.'''
        if self._params is None:
            return ValueError("Please run fit first")
        return self._params

    def fitfunc(self, *args, **kwargs):
        raise NotImplementedError

    def guess(self, *args, **kwargs):
        raise NotImplementedError


class VectorField2DLinearFitter(VectorField2DFitter):
    def __init__(self, params=None):
        ''' Fit a vector field to a linear model:
             [vx] =  [ gammaxx, gammaxy] . [x]
             [vy]    [ gammayx, gammayy]   [y]
        '''
        params = Parameters()
        params.add('gammaxx', 1)
        params.add('gammaxy', 0)
        params.add('gammayx', 0)
        params.add('gammayy', 1)

        super(VectorField2DLinearFitter, self).__init__(params=params)

    def fitfunc(self, x, y, gammaxx=0, gammaxy=0, gammayx=0, gammayy=0):
        ''' Fit function. Specify the matrix parameters for the fit.
            Matrix terms are:
             [vx] =  [ gammaxx, gammaxy] . [x]
             [vy]    [ gammayx, gammayy]   [y]
        '''
        mat = np.array([[gammaxx, gammaxy], [gammayx, gammayy]])
        r = np.array([x,y])
        return np.tensordot(mat, r, axes=(1,0))

    def guess(self, **kwargs):
        ''' No guess for this one. Just [1,0]
                                        [0,1]
        '''
        paramsdict = dict(gammaxx=1., gammaxy=0., gammayx=0., gammayy=1.)


        if kwargs is not None:
            for key in kwargs.keys():
                if key in paramsdict and key is not 'xy':
                    paramsdict[key] = kwargs[key]

        return paramsdict


class LineShape2DFitter:
    ''' Base class for all lineshape 2D Fitters.'''
    def __init__(self, params=None):
        ''' Initialize. If you set an initial guess
            this will be the static used guess function.
            If not, this will call the guess routine. 
            You need to implement the guess routine.
            
            Parameters
            ----------

            params : Parameters instance
                object specifying the default value and bounds of the
                parameters
        '''
        if params is None:
            raise ValueError("Sorry parameters not set, cannot continue")
        self.params = params

    def __call__(self, XY, img, **kwargs):
        ''' The call function will fit the function 
            img to the fit function specified in the object.

            Parameters
            ---------

            XY : np.ndarray, 3 dimensional
                the XY array for the [X,Y] coordinates of img

            kwargs : the initial guesses.

            Returns
            -------
            The best fit
        '''
        params = self.params.copy()
        guesskeys = self.guess(img, XY=XY)

        # make the parameters from the kwargs
        for key in self.params.keys():
            if key in kwargs.keys() and key is not 'XY':
                params[key].value = kwargs[key]
            else:
                # then guess
                params[key].value = guesskeys[key]

        self.mod = Model(self.fitfunc, independent_vars=['XY'],
                         param_names = self.params.keys())
        # assumes first var is dependent var
        res = self.mod.fit(img.ravel(), XY=(XY[0].ravel(),XY[1].ravel()), params=params,**kwargs)
        #add reduced chisq to parameter list
        res.best_values['chisq']=res.redchi
        return res.best_values
        
        
    def fitfunc(self):
        raise NotImplementedError

    def guess(self, img):
        raise NotImplementedError


class Gauss2DFitter(LineShape2DFitter):
    ''' A simple Gaussian 2D fitter.'''
    def __init__(self, **kwargs):
        ''' Initialize a Gaussian 2D Fitter object
            
            Parameters
            ---------
            kwargs : default arguments for fit
            in particular weights=1/errorbars
        '''
        params = self.init_parameters()
        super(Gauss2DFitter, self).__init__(params=params)

    def init_parameters(self, **kwargs):
        params = Parameters()
        params.add('baseline', value=0)
        params.add('amp', value=.15,min=0,max=.5)

        params.add('xc', value=11.,min=.0,max= 100.0)
        params.add('yc', value=11.,min=.0,max= 100.0)

        params.add('sigmax', value=1., min=1e-6, max=50.0)
        params.add('sigmay', value=1., min=1e-6, max=50.0)

        for key in kwargs.keys():
            if key in params:
                params[key].value = kwargs[key]

        return params


    def __call__(self, img, x=None, y=None, **kwargs):
        ''' fit for a Gaussian on the image, where x and y can be
        defined. If not defined, then guess'''
        if x is None:
            x = np.arange(img.shape[1])
        if y is None:
            y = np.arange(img.shape[0])

        XY = np.array(np.meshgrid(x,y))
            
        # doesn't make sense that the amplitude is negative here
        self.params['amp'].min = 0

        return super(Gauss2DFitter, self).__call__(XY, img,**kwargs)

    def fitfunc(self, XY, xc=None, yc=None, amp=1., baseline=0.,
                sigmax=1., sigmay=1.):
        ''' 
            xy : 2 by N by N matrix containing x and y
                xy[0] : x
                xy[1] : y

            xc, yc is center in (col, row) format, i.e. img[yc,xc]
        '''
        X = XY[0]
        Y = XY[1]

        if xc is None:
            xc = X.shape[1]//2

        if yc is None:
            yc = X.shape[0]//2

        return amp*np.exp(-(X-xc)**2/2./sigmax**2)*np.exp(-(Y-yc)**2/2./sigmay**2) + baseline

    def guess(self,img, XY=None,**kwargs):
        ''' Make a guess from the image of the Gaussian parameters.  Set
        the parameters with kwargs to bypass guessing for those specific
        parameters.
        
            Parameters
            ----------
            img : 2d np.ndarray
                the image to base the guess on

            XY : 3d np.ndarray
                the values for x (cols) and y (rows)
                x, y = XY
                default is to assume zero based integer numbering

            **kwargs : the keyword arguments to override guess with

            Returns
            -------
            paramsdict : dict
                dictionary of guesses
        '''
        # just guess image center
        paramsdict = dict()
        #yc, xc = np.array(img.shape)//2
        mx1 = np.argmax(img.ravel())
        xc, yc = mx1%img.shape[1], mx1//img.shape[0]

        if XY is not None:
            xc, yc = int(xc), int(yc)
            xc = np.minimum(np.maximum(0,xc),XY[0].shape[1]-1)
            yc = np.minimum(np.maximum(0,yc),XY[0].shape[0]-1)
            xc = XY[0][0, int(xc)]
            yc = XY[1][int(yc), 0]

        paramsdict['xc'] = xc 
        paramsdict['yc'] = yc 
        paramsdict['amp'] = img[int(yc), int(xc)]
        paramsdict['baseline'] = np.average(img)
        paramsdict['sigmax'] = 1 # make it one pixel in size
        paramsdict['sigmay'] = 1

        for key in kwargs.keys():
            if key in paramsdict and key is not 'xy':
                paramsdict[key] = kwargs[key]

        return paramsdict
