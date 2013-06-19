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


def alldata(response):
    return 'All data', 0, len(response.t)+1


def leadingedge(cutoff):
    def cutter(response):
        return 'Leading edge cutoff=%f' % cutoff, 0,  max(nonzero(response.y > cutoff*max(response.y))[0])
    return cutter


class heatmodel(object):
    def __init__(self, response, experiment, material, cutter=alldata):
        self.response = response
        self.cutname, self.startindex, self.endindex = cutter(response)
        self.L = float(experiment['Length/mm'])*1e-3

    def predictedresponse(self, parameters=None):
        if parameters is None: parameters = self.parameters

        kernel = self.fluxkernel(parameters)
        return convolve(self.response.u, -kernel)[:self.response.t.size]

    def fiterror(self, parameters):
        predictedy, kernelarea = self.predictedresponse(parameters)
        deviation = self.response.y - predictedy
        return linalg.norm(deviation[self.startindex:self.endindex])

    def fit(self):
        self.parameters = self.startparameters #scipy.optimize.minimize(self.fiterror, self.startparameters)


class analytic_zero(heatmodel):
    """ Analytic flux kernel - zero boundary conditions"""
    def __init__(self, response, experiment, material, cutter=alldata):
        super(analytic_zero, self).__init__(response, experiment, material, cutter=cutter)
        self.startparameters = [float(material['alpha'])/2, 0.1]

    def fluxkernel(self, parameters):
        Nterms = 100
        alpha, gain = parameters

        t = self.response.t
        L = self.L

        result = zeros_like(t)
        sgn = 1
        for n in range(1, Nterms):
            lambda_n = (2*n-1)*pi/(2*L)
            An = -(2./L)/lambda_n**2
            result -= where(t==0, 0,
                            2*alpha*lambda_n*exp(-alpha*lambda_n**2*t)*sgn/L)
            sgn *= -1
        return result*gain

    def label(self):
        return r'$u(L, t)=0$, $\alpha=%3.2f$ mm$^2$/s' % (self.parameters[0]*1e6)


class analytic_convec(heatmodel):

    def __init__(self, response, experiment, material, h, cutter=alldata):
        super(analytic_convec, self).__init__(response, experiment, material, cutter=cutter)
        self.startparameters = [float(material['alpha'])/3, h, 0.1]
        self.k = float(material['k'])

    def fluxkernel(self, parameters):
        Nterms = 100
        alpha, h, gain = parameters
        k = self.k
        L = self.L
        t = self.response.t

        def lambda_eq(lam):
            return lam*k/h*tan(lam*L) - 1

        result = zeros_like(t)
        lambda_n = -pi/L/2.0

        print alpha, h, gain, k, L

        for i in range(1, Nterms):
            lambda_n = scipy.optimize.newton(lambda_eq, lambda_n+pi/L*0.99)
            if not ( (i-1)*pi/L <= lambda_n <= i*pi/L ):
                print i, lambda_n

            An = -4*((L**2*h*lambda_n - (h - k)*L*lambda_n)*sin(L*lambda_n) - h*cos(L*lambda_n) + h)/(2*L*h*lambda_n**2 + h*lambda_n*sin(2*L*lambda_n))
            result += where(t==0, 0, 
                            An*alpha*lambda_n**3*exp(-alpha*lambda_n**2*t)*sin(lambda_n*L))

        return result

    def label(self):
        return r'Convective $\alpha=%3.2f$ mm$^2$/s, $h=%3.2f$ W/(m2.K)'  % (self.parameters[0]*1e6, self.parameters[1])


if __name__ == "__main__":
    db = fit.experimentdb(os.path.expanduser('~/Dropbox/Raw/experiments.csv'))
    materials = fit.materialdb(os.path.expanduser('~/Dropbox/Raw/material.csv'))
    for f in sys.argv[1:]:
        experiment = db.experiments[os.path.basename(f)]
        material = materials.index[experiment['Material']]
        r = fit.responsedata.fromlvm(f, int(experiment['Stride']))

        print f
        L = float(experiment['Length/mm'])
        alpha0, k0, rho0, Cp0 = map(float, (material[k] for k in
                                           ['alpha', 'k', 'rho', 'Cp']))
        alpha0 *= 1e-6

        print "Material:"
        print material


        # Do the fits - we can easily add other data ranges here.
        models  = [[analytic_zero(r, experiment, material), 'green'],
                   [analytic_convec(r, experiment, material, h=500), 'red']] 

        #plt.plot(r.t, r.y, color='blue', alpha=0.3, label='Data')

        for model, color in models:
            model.fit()

#             plt.plot(r.t, model.predictedresponse(),
#                      color=color, label=model.label())
            plt.plot(r.t, model.fluxkernel(model.parameters))


        plt.legend(loc='best')
        plt.title(r.name)
        #plt.savefig(f + '_heatkernel.png')
        plt.show()
        plt.cla()
