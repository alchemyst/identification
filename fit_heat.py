#!/usr/bin/env python

import lvm
import fit
from numpy import *
import matplotlib.pyplot as plt
import csv
import sys
import os.path


def fluxkernel(L, alpha, fluxmag, t, Nterms):
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


def normalize(signal):
    return signal - mean(signal[:10])


if __name__=="__main__":
    db = fit.experimentdb(os.path.expanduser('~/Dropbox/Raw/experiments.csv'))
    materials = fit.materialdb(os.path.expanduser('~/Dropbox/Raw/material.csv'))

    for f in sys.argv[1:]:
        experiment = db.experiments[os.path.basename(f)]
        d = lvm.lvm(f)
        r = fit.responsedata(d.data.X_Value,
                             normalize(d.data.Voltage_0),
                             normalize(d.data.Voltage),
                             name=f,
                             stride=int(experiment['Stride']))
        inputarea = trapz(r.u, r.t)
        outputarea = trapz(r.y, r.t)
        
        print f
        L = float(experiment['Length/mm'])/1000.
        alpha = float(materials.index[experiment['Material']]['Alpha'])
        kernel = fluxkernel(L, alpha, 1., r.t, 1000)
        predictedy = convolve(r.u, -kernel)[:r.t.size]
        kernelarea = trapz(predictedy, r.t)
        plt.plot(r.t, r.y, r.t, predictedy/kernelarea*outputarea)
        plt.legend(['Experimental', 'Analytic'], loc='best')
        plt.savefig(f + '_heatkernel.pdf')
        plt.cla()
