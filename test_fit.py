#!/usr/bin/env python

import unittest
import fit
import numpy
import scipy.signal

class testSystem(unittest.TestCase):
    def setUp(self):
        self.t = numpy.linspace(0, 10)
        self.u = numpy.ones_like(self.t)
        self.zeros = numpy.zeros_like(self.t)

    def testInit(self):
        G = fit.systemwithtimeconstants([], [1])


    def testResponse(self):

        G = fit.systemwithtimeconstants([], [1])

        r = fit.responsedata(self.t, self.u, self.u)

        sys = scipy.signal.lti(numpy.array([1]), numpy.array([1, 1]))
        rt, ry, _ = scipy.signal.lsim(sys, self.u, self.t)

        t, y = G.response(r)

        numpy.testing.assert_array_equal(ry, y)

class testResponseData(unittest.TestCase):
    def setUp(self):
        self.t = numpy.linspace(0, 10)
        self.u = numpy.ones_like(self.t)
        self.zeros = numpy.zeros_like(self.t)


    def testInit(self):
        y = self.u[:]

        r = fit.responsedata(self.t, self.u, y)

        numpy.testing.assert_array_equal(r.t, self.t)
        numpy.testing.assert_array_equal(r.u, self.u)
        numpy.testing.assert_array_equal(r.y, y)
        numpy.testing.assert_array_equal(r.du, self.zeros)
        numpy.testing.assert_array_equal(r.dy, self.zeros)



class testFitter(unittest.TestCase):
    def setUp(self):
        self.t = numpy.linspace(0, 10)
        self.u = numpy.ones_like(self.t)
        self.zeros = numpy.zeros_like(self.t)
        self.G0 = fit.systemwithtimeconstants([], [1])

    def testInit(self):
        y = self.u

        r = fit.responsedata(self.t, self.u, y)

        fitter = fit.fitter(r, self.G0)

    def testError(self):
        G = fit.systemwithtimeconstants([], [1])
        r = fit.responsedata(self.t, self.u, self.u)
        t, y = G.response(r)
        r = fit.responsedata(self.t, self.u, y)

        fitter = fit.fitter(r, G)
        self.assertAlmostEqual(fitter.error(), 0)

    def testEval(self):
        # construct first order response
        G = fit.systemwithtimeconstants([], [1])
        r = fit.responsedata(self.t, self.u, self.u)
        t, y = G.response(r)
        r = fit.responsedata(self.t, self.u, y)

        # Guess wrong time constant
        G0 = fit.systemwithtimeconstants([], [2])
        fitter = fit.fitter(r, G0)

        # Test that error is wrong with wrong guess
        self.assertNotAlmostEqual(fitter.error(), 0)

        # Test that we can evaluate with the right parameters
        self.assertAlmostEqual(fitter.evalparameters([1]), 0)


if __name__=="__main__":
    unittest.main()
