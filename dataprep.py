#!/usr/bin/env python
from ubc_AI.prepfold import pfd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.interpolate import  RectBivariateSpline as interp
from scipy import ndimage, array, ogrid, mgrid
import sys,os




if __name__ == '__main__':
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    if not (f1.endswith('pfd')):
        print 'file name %s not end with pfd ' % (f1)
        sys.exit(1)

    pfdfile = pfd(f1)
    pfdfile.dedisperse()
    profs = pfdfile.profs
    pshape = profs.shape
    x, y, z = profs.shape
    data = profs.reshape((-1,1))
    del pfdfile

    mean = np.mean(data)
    var = np.std(data)
    data = (data-mean)/var
    profs = data.reshape(pshape)
    X, Y, Z = ogrid[0:1:x,0:1:y,0:1:z]
    coords = array([X, Y, Z])
    coeffs = ndimage.spline_filter(profs )
    X, Y, Z = mgrid[0:1:8j,0:1:8j,0:1:8j]
    coords = array([X, Y, Z])
    newf = ndimage.map_coordinates(coeffs, coords, prefilter=False)

    np.save(f2, newf)
