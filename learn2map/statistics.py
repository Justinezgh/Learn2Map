from lenstools.image.convergence import ConvergenceMap
import astropy.units as u
import numpy as np

def power_spectrum(data, map_size, l_edges=np.arange(60.0, 2000.0, 15.0)):
    map = ConvergenceMap(data, angle=map_size*u.deg)
    nu, c = map.powerSpectrum(l_edges)
    
    return nu, c

def peak_counts(data, map_size, thresholds = None):
    map = ConvergenceMap(data, angle=map_size*u.deg)
    if thresholds is None:
        thresholds = np.arange(map.data.min(),map.data.max(),0.05)
    nu, peaks = map.peakCount(thresholds)

    return nu, peaks

def minkowski_functionals(data, map_size, thresholds=np.arange(-2.0,2.0,0.2)):
    map = ConvergenceMap(data, angle=map_size*u.deg)
    nu,V0,V1,V2 = map.minkowskiFunctionals(thresholds,norm=True)
    
    return nu,V0,V1,V2

def pdf(data, map_size, thresholds=None):
    map = ConvergenceMap(data, angle=map_size*u.deg)
    if thresholds is None:
        thresholds = np.arange(map.data.min(),map.data.max(),0.05)
    nu,p = map.pdf(thresholds)

    return nu, p