#!/usr/bin/env python

import lvm
import fit
from numpy import *
import matplotlib.pyplot as plt
import csv
import sys
import os.path
import scipy.optimize
import scipy.integrate
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
    def __init__(self, response, experiment, material, dofit=None, cutter=alldata):
        self.response = response
        self.cutname, self.startindex, self.endindex = cutter(response)
        self.L = float(experiment['Length/mm'])*1e-3
        self.k = float(material['k'])
        self.gain = response.outputarea/response.inputarea
        self.dofit = dofit

    def scale(self, parameters):
        return parameters/self.scalefactors

    def unscale(self, scaledparameters):
        return scaledparameters*self.scalefactors
        
    def predictedresponse(self, parameters=None):
        if parameters is None: parameters = self.parameters

        kernel = self.fluxkernel(parameters)
        # TODO: Figure out why this convolution has to be adjusted with the sampling time!
        return convolve(-self.response.u, kernel)[:self.response.t.size]*self.response.t[1]*self.gain

    def fiterror(self, parameters):
        predictedy = self.predictedresponse(parameters)
        deviation = self.response.y - predictedy
        return linalg.norm(deviation[self.startindex:self.endindex])

    def fit(self):
        if self.dofit is None:
            self.parameters = self.startparameters
        else:
            fitindexes = [self.parameternames.index(name) for name in self.dofit]
            scaledparameters = self.scale(self.startparameters)

            def fitfunction(scaledchangingparameters):
                scaledparameters[fitindexes] = scaledchangingparameters
                # FIXME: Overly simplistic constraint handling here.
                # penalize negative values
                if any(scaledparameters < 0):
                    return 100*linalg.norm(scaledparameters[scaledparameters<0])
                else:
                    return self.fiterror(self.unscale(scaledparameters))

            starting_scaled_changingparameters = scaledparameters[fitindexes]
            result = scipy.optimize.minimize(fitfunction, starting_scaled_changingparameters, options={'disp': True})

            scaledparameters[fitindexes] = result.x
            self.parameters = self.unscale(scaledparameters)

    def report(self):
        print "Model parameters:"
        for n, v in zip(self.parameternames, self.parameters):
            fixedorfitted = "fitted" if (self.dofit and n in self.dofit) else "fixed"
            print " {}: {} ({})".format(n, v, fixedorfitted)
        print "Experiment gain:", self.gain


class analytic_zero(heatmodel):
    """ Analytic flux kernel - zero boundary conditions"""
    def __init__(self, response, experiment, material, dofit=None, cutter=alldata):
        super(analytic_zero, self).__init__(response, experiment, material, dofit=dofit, cutter=cutter)
        self.startparameters = array([float(material['alpha'])])
        self.scalefactors = array([1e-6])
        self.parameternames = ['alpha']

    def fluxkernel(self, parameters=None):
        if parameters is None:
            parameters = self.parameters

        Nterms = 100
        epsilon = 1e-12

        alpha = parameters

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
        return result

    def label(self):
        return r'$u(L, t)=0$, $\alpha=%3.1f$ mm$^2$/s' % (self.parameters[0]*1e6)


class analytic_convec(heatmodel):

    def __init__(self, response, experiment, material, h, dofit=None, cutter=alldata):
        super(analytic_convec, self).__init__(response, experiment, material, dofit=dofit, cutter=cutter)
        self.startparameters = array([float(material['alpha']), h])
        self.scalefactors = array([1e-6, 1e3])
        self.parameternames = ['alpha', 'h']
        self.k = float(material['k'])


    def fluxkernel(self, parameters=None):
        Nterms = 100
        epsilon = 1e-10

        if parameters is None:
            parameters = self.parameters
            
        alpha, h = parameters
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

        return result

    def label(self):
        return r'Conv $\alpha=%3.1f\,\mathrm{mm}^2/\mathrm{s}, h=%3.1f\,\mathrm{W}/(\mathrm{m}^2\cdot \mathrm{K})$'  % (self.parameters[0]*1e6, self.parameters[1])


def loadfile(f):
    experiment = experiments.index[os.path.basename(f)]
    material = materials.index[experiment['Material']]
    response = fit.responsedata.fromlvm(f, int(experiment['Stride']))

    return experiment, material, response


def normalize(signal):
    return signal/max(signal)


def spacedmarks(x, y, Nmarks, data_ratio=None):
    if data_ratio is None:
        data_ratio = plt.gca().get_data_ratio()

    dydx = gradient(y, x[1])
    dxdx = gradient(x, x[1])*data_ratio

    arclength = scipy.integrate.cumtrapz(sqrt(dydx**2 + dxdx**2), x, initial=0)

    marks = linspace(0, max(arclength), Nmarks)
    markx = interp(marks, arclength, x)
    marky = interp(markx, x, y)

    return markx, marky


if __name__ == "__main__":
    materialplots = {}
    counter = 1
    dirname = os.path.expanduser('~/Dropbox/Raw')
    for f in sys.argv[1:]:
        print f
        experiment, material, response = loadfile(f)

        print "Material:"
        print material

        materialname = material['Material']
        if materialname not in materialplots:
            materialplots[materialname] = counter
            counter +=1 

        mfig = materialplots[materialname]

        # Do the fits - we can easily add other data ranges here.
        models  = [#[analytic_zero(response, experiment, material), 'green'],
        #[analytic_zero(response, experiment, material, dofit=True), 'yellow'],
                   [analytic_convec(response, experiment, material, h=float(experiment['h'])), 'green', True],
                   [analytic_convec(response, experiment, material, h=float(experiment['h']), dofit=['alpha']), 'red', False],
                   ]


        plotnumber = materialplots[materialname]
        plt.figure(plotnumber)
        plt.plot(response.t, response.y, label=experiment['Length/mm'] + ' mm')

        cfig = plt.figure(0)
        valueaxis = plt.subplot(2, 1, 1)
        valueaxis.set_xticklabels(())
        plt.subplots_adjust(hspace=0.001)
        plt.plot(response.t, response.y, color='blue', label='Data')
        markt, marky = spacedmarks(response.t, response.y, 30)
        plt.scatter(markt, marky, color='blue', s=10)
        plt.xlim(xmin=0)

        for model, color, overallplot in models:
            model.fit()
            predicted = model.predictedresponse()
            print "Input area:", response.inputarea
            print "Output area:", response.outputarea
            print "Kernel area:", trapz(-model.fluxkernel(model.parameters), response.t)
            print "Predicted area:", trapz(predicted, response.t)
            model.report()
            if overallplot:
                plt.figure(plotnumber)
                plt.plot(response.t, predicted, label=model.label())
            plt.figure(0)
            plt.subplot(2, 1, 1)
            plt.plot(response.t, predicted,
                     color=color, label=model.label())
            plt.subplot(2, 1, 2)
            plt.plot(response.t, response.y - predicted,
                     color=color, label=model.label())
            

        plt.subplot(2, 1, 1)
        plt.ylabel('Heat flux / $(W/m^2)$')
        plt.legend(loc='best')
        plt.title(response.name)
        plt.subplot(2, 1, 2)
        plt.ylabel('Residual (actual - predicted)') 
        plt.xlabel('Time / s')

        plt.savefig(f + '_heatkernel.png')
        #plt.show()
        plt.clf()
        
    for materialname, plotnumber in materialplots.iteritems():
        plt.figure(plotnumber)
        plt.legend()
        plt.title(materialname)
        plt.savefig(os.path.join(dirname, materialname + '.png'))