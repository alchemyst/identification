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

experiments = fit.experimentdb(os.path.expanduser('~/Dropbox/Raw/experiments.csv'))
materials = fit.materialdb(os.path.expanduser('~/Dropbox/Raw/material.csv'))    


def alldata(response):
    return 'All data', 0, len(response.t)+1


def leadingedge(cutoff):
    def cutter(response):
        return 'Leading edge cutoff=%f' % cutoff, 0,  max(nonzero(response.y > cutoff*max(response.y))[0])
    return cutter


class heatmodel(object):
    """ Generic heat model class, based on fitting an experiment """
    def __init__(self, response, experiment, material, dofit=False, cutter=alldata):
        self.response = response
        self.cutname, self.startindex, self.endindex = cutter(response)
        self.L = float(experiment['Length/mm'])*1e-3
        self.k = float(material['k'])
        self.dofit = dofit

    def scale(self, parameters):
        return parameters/self.scalefactors

    def unscale(self, scaledparameters):
        return scaledparameters*self.scalefactors
        
    def predictedresponse(self, parameters=None):
        if parameters is None: parameters = self.parameters

        kernel = self.fluxkernel(parameters)
        return convolve(-self.response.u, kernel)[:self.response.t.size]

    def fiterror(self, scaledparameters):
        predictedy = self.predictedresponse(self.unscale(scaledparameters))
        deviation = self.response.y - predictedy
        return linalg.norm(deviation[self.startindex:self.endindex])

    def fit(self):
        if self.dofit:
            result = scipy.optimize.minimize(self.fiterror, self.scale(self.startparameters), options={'disp': True})
            self.parameters = self.unscale(result.x)
        else:
            self.parameters = self.startparameters

    def report(self):
        for n, v in zip(self.parameternames, self.parameters):
            print "{}: {}".format(n, v)


class analytic_zero(heatmodel):
    """ Analytic flux kernel - zero boundary conditions"""
    def __init__(self, response, experiment, material, dofit=False, cutter=alldata):
        super(analytic_zero, self).__init__(response, experiment, material, dofit=dofit, cutter=cutter)
        self.startparameters = array([float(material['alpha']), 1])
        self.scalefactors = array([1e-6, 1])
        self.parameternames = ['alpha', 'gain']
        
    def fluxkernel(self, parameters):
        Nterms = 100
        epsilon = 1e-10
        
        alpha, gain = parameters

        t = self.response.t
        L = self.L

        result = zeros_like(t)
        sgn = 1
        for n in range(1, Nterms):
            lambda_n = (2*n-1)*pi/(2*L)
            #An = -(2./L)/lambda_n**2
            term = where(t==0, 0,
                         2*alpha*lambda_n*exp(-alpha*lambda_n**2*t)*sgn/L)
            if max(abs(term)) < epsilon: 
                break
            result -= term
            sgn *= -1
        return result*gain

    def label(self):
        return r'$u(L, t)=0$, $\alpha=%3.1f$ mm$^2$/s' % (self.parameters[0]*1e6)


class analytic_convec(heatmodel):

    def __init__(self, response, experiment, material, h, dofit=False, cutter=alldata):
        super(analytic_convec, self).__init__(response, experiment, material, dofit=dofit, cutter=cutter)
        self.startparameters = array([float(material['alpha']), h, 1])
        self.scalefactors = array([1e-6, 1e3, 1])
        self.parameternames = ['alpha', 'h', 'gain']
        self.k = float(material['k'])


    def fluxkernel(self, parameters):
        Nterms = 100
        epsilon = 1e-10

        alpha, h, gain = parameters
        k = self.k
        L = self.L
        t = self.response.t

        def lambda_eq(lam):
            return sin(lam*L) - h/k*cos(lam*L)/lam

        result = zeros_like(t)

        leftlim = 0
        rightlim = pi/L/2.

        for i in range(1, Nterms):
            lambda_n = scipy.optimize.ridder(lambda_eq, leftlim, rightlim)
            leftlim, rightlim = rightlim, rightlim + pi/L

            assert ( (i-1)*pi/L <= lambda_n <= i*pi/L ), 'Lambda out of sequence'
            An = -4/(2*L*lambda_n**2 + lambda_n*sin(2*L*lambda_n))
            term = where(t==0, 0,
                         An*alpha*lambda_n**3*exp(-alpha*lambda_n**2*t)*sin(lambda_n*L))
            if max(abs(term)) < epsilon:
                break
            result += term

        return result*gain

    def label(self):
        return r'Conv $\alpha=%3.1f$ mm$^2$/s, $h=%3.1f$ W/(m2.K)'  % (self.parameters[0]*1e6, self.parameters[1])

def loadfile(f):
    experiment = experiments.index[os.path.basename(f)]
    material = materials.index[experiment['Material']]
    response = fit.responsedata.fromlvm(f, int(experiment['Stride']))
    
    return experiment, material, response

def normalize(signal):
    return signal/max(signal)

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print f
        experiment, material, response = loadfile(f)

        print "Material:"
        print material

        # Do the fits - we can easily add other data ranges here.
        models  = [[analytic_zero(response, experiment, material), 'green'],
        #[analytic_zero(response, experiment, material, dofit=True), 'yellow'],
                   [analytic_convec(response, experiment, material, h=float(experiment['h'])), 'magenta'],
        #[analytic_convec(response, experiment, material, h=float(experiment['h']), dofit=True), 'red'],
                   ]

        plt.plot(response.t, response.y, color='blue', alpha=0.3, label='Data')

        for model, color in models:
            model.fit()
            model.report()
            predicted = model.predictedresponse()
            gain = response.inputarea/response.outputarea
            print "Input area:", response.inputarea
            print "Output area:", response.outputarea
            print "Gain:", gain
            print "Kernel area:", trapz(-model.fluxkernel(model.parameters), response.t)
            print "Predicted area:", trapz(predicted, response.t)
            plt.plot(response.t, predicted*gain,
                     color=color, label=model.label())

        plt.ylabel('Heat flux')
        plt.xlabel('Time / s')
        plt.legend(loc='best')
        plt.title(response.name)
        #plt.savefig(f + '_heatkernel.png')
        plt.show()
        plt.cla()
