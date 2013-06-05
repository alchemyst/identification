#!/usr/bin/env python

import lvm
import fit
from numpy import *
import matplotlib.pyplot as plt
import csv
import sys
import os.path
import scipy.optimize
from functools import partial

def fluxkernel(L, alpha, fluxmag, t, Nterms=1000):
    """ Analyitic flux kernel"""
    result = zeros_like(t)
    sgn = 1
    for n in range(1, Nterms):
        lambda_n = (2*n-1)*pi/(2*L)
        An = -(2./L)/lambda_n**2
        result -= where(t==0, 0,
                        2*alpha*lambda_n*exp(-alpha*lambda_n**2*t)*sgn/L)
        sgn *= -1
    return result*fluxmag


def predictedresponse(response, L, alpha, gain=1, Nterms=1000):
    kernel = fluxkernel(L, alpha, 1., response.t, Nterms)
    predictedy = convolve(response.u, -kernel)[:response.t.size]
    kernelarea = trapz(predictedy, response.t)

    return predictedy*gain, kernelarea*gain


def fiterror(x, L, response, startindex=None, endindex=None):
    alpha, gain = x
    if startindex is None:
        startindex = 0
    if endindex is None:
        endindex = len(response.t)+1
    predictedy, kernelarea = predictedresponse(response, L, alpha, gain)
    deviation = response.y - predictedy
    return linalg.norm(deviation[startindex:endindex])

def alphalabel(alpha):
    return r'Analytic, $\alpha=%3.2f$ mm$^2$/s' % (alpha)

if __name__=="__main__":
    db = fit.experimentdb(os.path.expanduser('~/Dropbox/Raw/experiments.csv'))
    materials = fit.materialdb(os.path.expanduser('~/Dropbox/Raw/material.csv'))

    for f in sys.argv[1:]:
        experiment = db.experiments[os.path.basename(f)]
        r = fit.responsedata.fromlvm(f, int(experiment['Stride']))

        print f
        L = float(experiment['Length/mm'])
        alpha0 = float(materials.index[experiment['Material']]['Alpha'])*1e6

        print "Material alpha:", alpha0

        startindex = 0
        cutoff = 0.95
        endindex = max(nonzero(r.y > cutoff*max(r.y))[0])

        x0 = [alpha0/2., 0.1]
        
        # Do the fits - we can easily add other data ranges here.        
        fits  = [['Full dataset', 0, len(r.t), 'blue',
                  scipy.optimize.minimize(fiterror, x0, args=(L, r))],
                 ['Leading edge', startindex, endindex, 'red', 
                  scipy.optimize.minimize(fiterror, x0, args=(L, r, startindex, endindex))],
                 ]
        
        for name, startindex, endindex, color, result in fits:
            alpha, gain = result.x
            print name
            print "Fitted alpha:", alpha
            print "Fitted gain:", gain

            predictedy, kernelarea = predictedresponse(r, L, alpha, gain=gain)
            plt.plot(r.t[startindex:endindex], r.y[startindex:endindex], color=color, alpha=0.3, label=name)
            plt.plot(r.t, predictedy, color=color, label=alphalabel(alpha))

        plt.legend(loc='best')
        plt.title(r.name)
        plt.savefig(f + '_heatkernel.png')
        plt.cla()
