#!/usr/bin/env python

import pandas
import numpy
import scipy.signal as sig
import matplotlib.pyplot as plt

#
#   u    +----------+   y
# ------>|    G     |-------->
#        +----------+
#


filename = 'heat_flux.csv'

generate = False

def timeconstants(taus):
    r = [1]
    for tau in taus:
        r = numpy.convolve(r, [tau, 1])
    return r


def systemwithtimeconstants(tau_num, tau_den):
    return sig.lti(timeconstants(tau_num), timeconstants(tau_den))


class responsedata:
    def __init__(self, t, u, y, name):
        # Some error checking for the unwary
        assert numpy.linalg.norm(numpy.diff(t, 2)) < 1e-9, "Sampling period must be constant"

        # sampling period: first time step
        self.T = t[1]

        self.t = t
        self.u = u
        self.y = y
        self.name = name
        self.du = numpy.gradient(u)/self.T
        self.dy = numpy.gradient(y)/self.T
    

class fitted:
    def __init__(self, data, G):
        self.data = data
        self.G = G

    def response(self):
        Gt, Gy, _ = sig.lsim(self.G, self.data.u, self.data.t)
        return Gt, Gy


def split(x, point):
    return (x[:point], x[point:])


def compare(sys, w_cutoff, gainadjustment=1, timefactor=1):
    Gt, Gy = sys.response()

    # obtain frequency response of transfer function
    w_tf, Gw_tf = sig.freqs(sys.G.num, sys.G.den)

    # Frequency response of input - diff for derivative if steps
    duw = numpy.fft.fft(sys.data.du)
    dyw = numpy.fft.fft(sys.data.dy)

    # Frequency response of output (division is like deconvolution)
    Gw = dyw/duw

    # Find what frequencies the FFT was for
    N = Gw.size
    w = numpy.fft.fftfreq(N, d=sys.data.T)*2*numpy.pi

    # Don't use the negative frequencies for Bode diagram
    useful = w >= 0
    w_useful = w[useful]
    Gw_useful = Gw[useful]

    # Filter out frequencies above the cutoff
    Gw_filtered = Gw*(numpy.abs(w)<w_cutoff)

    # Visualise

    def plotlines():
        plt.axvline(w_cutoff)
        for p in -sys.G.poles:
            plt.axvline(p, color='r')
        for z in -sys.G.zeros:
            plt.axvline(z, color='g')

    
    plt.subplot(3, 1, 1)
    plt.loglog(w_tf, numpy.abs(Gw_tf),
               w_useful, numpy.abs(Gw_useful)*gainadjustment)
    plt.title(sys.data.name)
    plotlines()
    plt.xlabel('Frequency (rad/sec)')
    plt.ylabel('Magnitude')
    plt.legend(['Analytic', 'FFT'], 'best')

    plt.subplot(3, 1, 2)
    plt.semilogx(w_tf, numpy.unwrap(numpy.angle(Gw_tf)),
                 w_useful, numpy.unwrap(numpy.angle(Gw_useful)))
    plotlines()
    plt.xlabel('Frequency (rad/sec)')
    plt.ylim([-5, 0])
    plt.yticks(-numpy.arange(4)*numpy.pi/2)
    plt.grid()
    plt.ylabel('Phase')

    # Find system impulse response (time domain version of transfer function)
    fftimpulse = numpy.real(numpy.fft.ifft(Gw_filtered))*gainadjustment
    # Find system response to input, taking only first bit
    fftresponse = sig.convolve(sys.data.u, fftimpulse)[:sys.data.t.size]
    plt.subplot(3, 1, 3)

    plt.plot(sys.data.t, sys.data.y,
             Gt*timefactor, Gy,
             sys.data.t, fftresponse )
    plt.legend(['Data', 'Analytic', 'FFT'], 'best')

    plt.savefig(sys.data.name + '.pdf')

if __name__ == "__main__":
    if generate:
        # Write to csv
        time = pandas.Series(t, name='t')
        pandas.DataFrame({'u': u, 'y': y}, index=time).to_csv(filename)

    # load data from csv file
    data = numpy.recfromcsv(filename)[:5000]

    # Create step input
    u = numpy.ones_like(data.t)
    u[0] = 0
    plt.figure()

    r1 = responsedata(data.t, u, data.y1, 'u-y1')
    r2 = responsedata(data.t, u, data.y2, 'u-y2')
    r3 = responsedata(data.t, data.y1, data.y2, 'y1-y2')

    G1 = fitted(r1, systemwithtimeconstants([4, 4, 180], [1430, 80, 80]))
    G2 = fitted(r2, systemwithtimeconstants([4, 180], [1430, 80, 80, 140]))
    G3 = fitted(r3, systemwithtimeconstants([4], [140]))

    compare(G1, 0.1, gainadjustment=1.0/0.67, timefactor=0.5)
    plt.figure()
    compare(G2, 2e-2, gainadjustment=1.0/0.67, timefactor=0.5)
    plt.figure()
    compare(G3, 2e-2, gainadjustment=1.0/0.9)
    #plt.show()
