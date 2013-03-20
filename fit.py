#!/usr/bin/env python

import pandas
import numpy
import scipy.signal as sig
import scipy.optimize
import matplotlib.pyplot as plt
import copy

#
#   u    +----------+   y
# ------>|    G     |-------->
#        +----------+
#


filename = 'heat_flux.csv'

def timeconstants(taus):
    r = [1]
    for tau in taus:
        r = numpy.convolve(r, [tau, 1])
    return r

class responsedata:
    """ Container for the response data of an experiment.
    """

    @staticmethod
    def fromfile(filename):
        d = numpy.recfromcsv(filename)
        return responsedata(d.t, d.u, d.y, filename + 'u-y')

    def __init__(self, t, u, y, name=None):
        # Some error checking for the unwary
        assert numpy.linalg.norm(numpy.diff(t, 2)) < 1e-9, "Sampling period must be constant"

        # sampling period: first time step
        self.T = t[1]

        self.t = t
        self.u = u
        self.y = y
        if name is not None:
            self.name = name
        else:
            self.name = "u-y"
        self.du = numpy.gradient(u)/self.T
        self.dy = numpy.gradient(y)/self.T

    def response(self, data):
        return self.t, self.y

    def plotresponse(self):
        plt.plot(self.t, self.u, self.t, self.y)
        plt.legend(['u', 'y'])

class systemwithtimeconstants:
    def __init__(self, tau_num, tau_den, timeadjustment=1):
        self.tau_num = tau_num[:]
        self.tau_den = tau_den[:]
        self.G = sig.lti(timeconstants(tau_num), timeconstants(tau_den))
        # obtain frequency response of transfer function
        self.w_tf, self.Gw_tf = sig.freqs(self.G.num, self.G.den)
        self.timeadjustment = timeadjustment

    def response(self, data):
        Gt, Gy, _ = sig.lsim(self.G, data.u, data.t)
        return Gt*self.timeadjustment, Gy

    def bodemag(self):
        plt.loglog(self.w_tf, numpy.abs(self.Gw_tf))
        plt.xlabel('Frequency (rad/sec)')
        plt.ylabel('Magnitude')
        self.plotlines()

    def bodephase(self):
        plt.semilogx(self.w_tf, numpy.unwrap(numpy.angle(self.Gw_tf)))
        plt.xlabel('Frequency (rad/sec)')
        plt.ylabel('Phase')
        self.plotlines()

    def plotlines(self):
        for p in -self.G.poles:
            plt.axvline(p, color='r')
        for z in -self.G.zeros:
            plt.axvline(z, color='g')

    def __repr__(self):
        return "systemwithtimeconstants(%s, %s)" % (self.tau_num, self.tau_den)


class fitter:
    def __init__(self, data, initialsystem):
        self.data = data
        self.G0 = initialsystem
        self.G = copy.copy(self.G0)
        self.calcresponse()

    def error(self):
        return numpy.linalg.norm(self.data.y - self.y)

    def calcresponse(self):
        self.t, self.y = self.G.response(self.data)

    def gensystem(self, x):
        N = len(self.G0.tau_num)

        return systemwithtimeconstants(x[:N], x[N:])

    def evalparameters(self, x):
        self.G = self.gensystem(x)
        self.calcresponse()

        return self.error()

    def fit(self):
        x0 = self.G0.tau_num + self.G0.tau_den
        xopt = scipy.optimize.fmin(self.evalparameters, x0)
        self.G = self.gensystem(xopt)


class fft:
    """ class for handling the frequency response based on FFT """
    def __init__(self, data, w_cutoff, gainadjustment=1, deriv=False):
        self.data = data
        self.w_cutoff = w_cutoff
        self.gainadjustment = gainadjustment
        self.deriv = deriv
        self.calc()

    def calc(self):
        if self.deriv:
            self.duw = numpy.fft.fft(self.data.du)
            self.dyw = numpy.fft.fft(self.data.dy)
            # Frequency response of output (division is like deconvolution)
            self.Gw = self.dyw/self.duw
        else:
            self.uw = numpy.fft.fft(self.data.u)
            self.yw = numpy.fft.fft(self.data.y)
            # Frequency response of output (division is like deconvolution)
            self.Gw = self.yw/self.uw

        # Find what frequencies the FFT was for
        self.N = self.Gw.size
        self.w = numpy.fft.fftfreq(self.N, d=self.data.T)*2*numpy.pi

        # Don't use the negative frequencies for Bode diagram
        self.useful = self.w >= 0
        self.w_useful = self.w[self.useful]
        self.Gw_useful = self.Gw[self.useful]

        # Filter out frequencies above the cutoff
        self.Gw_filtered = self.Gw*(numpy.abs(self.w)<self.w_cutoff)

        # Find system impulse response (time domain version of transfer function)
        self.impulse = numpy.real(numpy.fft.ifft(self.Gw_filtered))*self.gainadjustment

    def response(self, data):
        # Find system response to input, taking only first bit
        return data.t, sig.convolve(data.u, self.impulse)[:data.t.size]


    def bodemag(self):
        plt.loglog(self.w_useful, numpy.abs(self.Gw_useful)*self.gainadjustment)
        plt.xlabel('Frequency (rad/sec)')
        plt.ylabel('Magnitude')
        self.plotlines()

    def bodephase(self):
        plt.semilogx(self.w_useful, numpy.unwrap(numpy.angle(self.Gw_useful)))
        plt.xlabel('Frequency (rad/sec)')
        plt.ylabel('Phase')
        self.plotlines()

    def plotlines(self):
        plt.axvline(self.w_cutoff)


def compare(data, sys, fft):
    # Magnitudes
    plt.subplot(3, 1, 1)
    sys.bodemag()
    fft.bodemag()
    plt.title(data.name)
    plt.legend(['Analytic', 'FFT'], 'best')

    # Phases
    plt.subplot(3, 1, 2)
    sys.bodephase()
    fft.bodephase()
    plt.ylim([-5, 0])
    plt.yticks(-numpy.arange(4)*numpy.pi/2)
    plt.grid()

    # Responses
    plt.subplot(3, 1, 3)
    for thing in [sys, fft, data]:
        plt.plot(*thing.response(data))
    plt.legend(['Analytic', 'FFT', 'Data'], 'best')


if __name__ == "__main__":
    # load data from csv file
    data = numpy.recfromcsv(filename)[:5000]

    # Create step input
    u = numpy.ones_like(data.t)
    u[0] = 0
    plt.figure()

    rs = [responsedata(data.t, u, data.y1, 'u-y1'),
          responsedata(data.t, u, data.y2, 'u-y2'),
          responsedata(data.t, data.y1, data.y2, 'y1-y2'),
          ]

    Gs = [systemwithtimeconstants([4, 4, 180], [1430, 80, 80], 0.5),
          systemwithtimeconstants([4, 180], [1430, 80, 80, 140], 0.5),
          systemwithtimeconstants([4], [140], 1),
          ]

    ffts = [fft(rs[0], 0.1, 1.0/0.67, True),
            fft(rs[1], 2e-2, 1.0/0.67, True),
            fft(rs[2], 2e-2, 1.0/0.9, True),
            ]

    for r, G, fft in zip(rs, Gs, ffts):
        plt.figure()
        compare(r, G, fft)
        plt.savefig(r.name + '.pdf')

    #plt.show()
