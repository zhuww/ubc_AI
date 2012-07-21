import numpy as np
import os, glob

def normalize(data):
    '''data:input array of 1-3 dimentions
       to be normalized.
       Remember to return the normalized data. 
       The input will not be changed.
    '''
    if type(data) in [list]:
        result = []
        for a in data:
            result.append(normalize(a))
        return result
    else:
        shape = data.shape
        data = data.reshape((-1,1))
        mean = np.mean(data)
        var = np.std(data)
        data = (data-mean)/var
        data = data.reshape(shape)
        return data

from scipy.interpolate import RectBivariateSpline as interp2d
from scipy import ndimage, array, ogrid, mgrid

def downsample(a, n):
    '''a: input array of 1-3 dimentions
       n: downsample to n bins
    '''
    if type(a) in [list]:
        result = []
        for b in a:
            result.append(downsample(b))
        return result
    else:
        shape = a.shape
        D = len(shape)
        if D == 1:
            coords = mgrid[0:1:1j*n]
        elif D == 2:
            d1,d2 = shape
            coords = mgrid[0:d1:1j*n, 0:d2:1j*n]
        elif D == 3:
            d1,d2,d3 = shape
            coords = mgrid[0:d1:1j*n, 0:d2:1j*n, 0:d3:1j*n]
        else:
            raise "too many dimentions %s " % D
        def map_to_index(x,bounds,N):
            xmin, xmax= bounds
            return (x - xmin)/(xmax-xmin)*N
        if D == 1:
            m = len(a)
            x = mgrid[0:1:1j*m]
            #newf = interp(x, a, bounds_error=True)
            #return newf(coords)
            return np.interp(coords, x, a)
        elif D == 2:
            #k,l = a.shape
            #x = mgrid[0:1:1j*k]
            #y = mgrid[0:1:1j*l]
            #f = interp2d(x, y, a)
            #coords = mgrid[0:1:1j*n]
            #return f(coords, coords)
            newf = ndimage.map_coordinates(a, coords)
            return newf
        else:
            #coeffs = ndimage.spline_filter(a)
            newf = ndimage.map_coordinates(coeffs, coords, prefilter=False)
            #newf = ndimage.map_coordinates(coeffs, coords )
            return newf

import glob
from prepfold import pfd
SAMPLE_FILES_DIR = '/data/pulse-learning/Erik/'
def load_pfds(dir=SAMPLE_FILES_DIR):
    SAMPLE_FILES = glob.glob(dir+'*.pfd')
    pfds = []
    for f in SAMPLE_FILES:
        pf = pfd(f)
        pfds.append(pf)
    return pfds

def extractdata(pfds, d, normalize=False, downsample=0):
    """d in [1,2,3]"""
    if not d in [1,2,3]:
        raise "d must be in [1,2,3], but assigned %s" % d
    data = []
    for pf in pfds:
        pf.dedisperse()
        profile = pf.profs
        D = len(profile.shape)
        i = D - d
        if i == 1:
            profile = profile.sum(0)
        elif i == 2:
            profile = profile.sum(0).sum(0)
        data.append(profile)
    #data = np.ndarray(data)

    return data

def load_samples(*args, **kws):
    return extractdata(load_pfds(), *args, **kws)

def quick_load_samples(*args, **kws):
    #if os.access(SAMPLE_FILES_DIR+"samples.npy", os.R_OK):
    samples = []
    for sf in glob.glob(SAMPLE_FILES_DIR+"samples_*.npy"):
        profile = np.load(sf)
        D = len(profile.shape)
        #print type(samples), samples.shape
        if len(args) > 0 and args[0] < 3:
            i = D - args[0]
            if i == 1:
                profile = profile.sum(0)
            elif i == 2:
                #profile = profile.sum(1).T.sum(1)
                profile = profile.sum(0).sum(0)
        samples.append(profile)
    return samples

    #else:
        #return extractdata(load_pfds(), *args, **kws)

if __name__ == '__main__':
    samples = load_samples(3)
    for i,s in enumerate(samples):
        np.save(SAMPLE_FILES_DIR+'samples_%s' % i, s)
    #print quick_load_samples(1)[0].shape



