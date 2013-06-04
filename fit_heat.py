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


def predictedresponse(response, L, alpha, Nterms=1000):
    kernel = fluxkernel(L, alpha, 1., response.t, Nterms)
    predictedy = convolve(response.u, -kernel)[:response.t.size]
    kernelarea = trapz(predictedy, response.t)

    return predictedy/kernelarea*response.outputarea, kernelarea


def fiterror(alpha, L, response, startindex=None, endindex=None):
    if startindex is None:
        startindex = 0
    if endindex is None:
        endindex = len(response.t)+1
    predictedy, kernelarea = predictedresponse(response, L, alpha)
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
        predictedy0, kernelarea0 = predictedresponse(r, L, alpha0)
        print "Material alpha:", alpha0
        #fitresult = scipy.optimize.minimize_scalar(fiterror, bounds=[alpha0/10., alpha0], args=(L, r), method="golden")
        # alpha = scipy.optimize.golden(fiterror, brack=(alpha0/20., alpha0/10, alpha0), args=(L, r))
        startindex = 0
        cutoff = 0.5
        endindex = max(nonzero(r.y > cutoff*max(r.y))[0])
        alpha = scipy.optimize.fminbound(fiterror, alpha0/30., alpha0/2.,
                                         args=(L, r, startindex, endindex))

        #alpha = fitresult.x
        #alpha = alpha0/20
        print "Fitted alpha:", alpha
        predictedy, kernelarea = predictedresponse(r, L, alpha)
        plt.plot(r.t, r.y,
                 r.t, predictedy0,
                 r.t, predictedy)
        plt.axvline(r.t[endindex])
        plt.axhline(cutoff*max(r.y), alpha=0.3)
        plt.legend(['Experimental',
                    alphalabel(alpha0),
                    alphalabel(alpha)], loc='best')
        plt.title(r.name)
        plt.savefig(f + '_heatkernel.pdf')
        plt.cla()
