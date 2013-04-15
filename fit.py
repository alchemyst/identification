#!/usr/bin/env python

import numpy
import scipy.signal as sig
import scipy.optimize
import matplotlib.pyplot as plt
import copy
import lvm

#
#   u    +----------+   y
# ------>|    G     |-------->
#        +----------+
#

def timeconstants(taus, gain=1):
    r = [gain]
    for tau in taus:
        r = numpy.convolve(r, [tau, 1])
    return r


class responsedata:
    """ Container for the response data of an experiment.
    """

    @staticmethod
    def fromfile(filename):
        """ Factory method: Create a responsedata object from a filename or file object
        """
        d = numpy.recfromcsv(filename)
        name = filename.name if hasattr(filename, 'name') else filename
        return responsedata(d.t, d.u, d.y, name + 'u-y')

    def __init__(self, t, u, y, name=None):
        # Some error checking for the unwary
        assert numpy.linalg.norm(numpy.diff(t, 2)) < 1e-9, \
               "Sampling period must be constant"

        # sampling period: first time step
        self.T = t[1]

        self.t = t
        self.u = u
        self.y = y
        if name is not None:
            self.name = name
        else:
            self.name = "u-y"
        self.du = numpy.gradient(u) / self.T
        self.dy = numpy.gradient(y) / self.T

    def response(self, data):
        return self.t, self.y

    def plotresponse(self):
        plt.plot(self.t, self.u, self.t, self.y)
        plt.legend(['u', 'y'])


class systemwithtimeconstants:
    def __init__(self, tau_num, tau_den, gain=1, timeadjustment=1, Dt=0):
        self.tau_num = tau_num[:]
        self.tau_den = tau_den[:]
        self.gain = gain
        self.Dt = Dt
        self.G = sig.lti(timeconstants(tau_num, gain), timeconstants(tau_den))
        # obtain frequency response of transfer function
        self.w_tf, self.Gw_tf = sig.freqs(self.G.num, self.G.den)
        self.Gw_tf *= numpy.exp(-self.w_tf*1j*self.Dt)
        self.timeadjustment = timeadjustment

    def response(self, data):
        Gt, Gy, _ = sig.lsim(self.G, data.u, data.t)
        Gy = numpy.interp(Gt - self.Dt, Gt, Gy)
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
        return "systemwithtimeconstants(%s, %s, gain=%s, Dt=%s)" % (self.tau_num, self.tau_den, self.gain, self.Dt)


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

        return systemwithtimeconstants(tau_num=x[:N],
                                       tau_den=x[N:-2],
                                       gain=x[-2],
                                       Dt=x[-1])

    def evalparameters(self, x):
        self.G = self.gensystem(x)
        self.calcresponse()

        return self.error()

    def fit(self):
        x0 = self.G0.tau_num + self.G0.tau_den + [self.G0.gain, self.G0.Dt]
        xopt = scipy.optimize.fmin(self.evalparameters, x0)
        self.G = self.gensystem(xopt)


class fft:
    """ class for handling the frequency response based on FFT """
    def __init__(self, data, w_cutoff, gainadjustment=1, deriv=False):
        """ Note this is not the optimal way to calculate an approximate transfer function frequency response """

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
        self.Gw_filtered = self.Gw*(numpy.abs(self.w) < self.w_cutoff)

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
    plt.subplot(3, 2, 5)
    plt.plot(data.t, data.u)
    for thing in [sys, fft, data]:
        plt.plot(*thing.response(data))
    plt.legend(['Input', 'Analytic', 'FFT', 'Data'], 'best')

    plt.subplot(3, 2, 6)
    plt.plot(data.t, scipy.integrate.cumtrapz(data.u, initial=0))
    for thing in [sys, fft, data]:
        t, y = thing.response(data)
        plt.plot(t, scipy.integrate.cumtrapz(y, initial=0))


if __name__ == "__main__":
    # load data from csv file
    import argparse
    parser = argparse.ArgumentParser(description='Fit linear model to impulse data')
    parser.add_argument('datafiles', nargs='+', type=argparse.FileType('r'),
                        help='Filenames to use in fit. Must be CSVs with three columns with labels t, u and y.')
    parser.add_argument('--num', default=[], nargs='+', type=float,
                        help='Numerator time constants')
    parser.add_argument('--den', default=[20], nargs='+', type=float,
                        help='Denominator time constants')
    parser.add_argument('--dt', default=0, nargs=1, type=float,
                        help='Dead time')
    parser.add_argument('--fit', default=False, action='store_true',
                        help='Run fitting routine to improve fit')
    parser.add_argument('--cutoff', default=1e-1, type=float,
                        help='Frequency cutoff for fft')
    parser.add_argument('--starttime', default=0, nargs=1, type=int,
                        help='Remove data before this time and re-stamp')
    parser.add_argument('--endtime', default=numpy.Inf, nargs=1, type=int,
                        help='Remove data after this time')
    parser.add_argument('--selectgood', default=False, action='store_true',
                        help='Select good data interactively')
    parser.add_argument('--save', default=False, action='store_true',
                        help='Save results to .pdf')
    args = parser.parse_args()

    for f in args.datafiles:
        if f.name.lower().endswith('csv'):
            data = responsedata.fromfile(f)
        elif f.name.lower().endswith('lvm'):
            l = lvm.lvm(f)
            data = responsedata(l.data.X_Value, l.data.Voltage_0, l.data.Voltage, name=f.name)
            data.u -= data.u.min()
            data.y -= data.y.min()

        if args.selectgood:
            data.plotresponse()
            ((args.starttime, _), (args.endtime, _)) = plt.ginput(2)

        plt.figure()

        good = (data.t >= args.starttime) & (data.t <= args.endtime)
        data.t -= args.starttime
        data.t = data.t[good]
        data.u = data.u[good]
        data.y = data.y[good]
                
        G = systemwithtimeconstants(args.num, args.den, Dt=args.dt[0])
        thefft = fft(data, args.cutoff)
        if args.fit:
            thefitter = fitter(data, G)
            thefitter.fit()
            G = thefitter.G
        
        print G

        compare(data, G, thefft)
        if args.save:
            plt.savefig(data.name + '.pdf')

    if not args.save:
        plt.show()
