Simple identification
=====================

fit.py supplies classes and functions for simple identification of
transfer functions from pulse test data.

Simple usage:

    python fit.py filename.csv

This will use the FFT to calculate the frequency response of the
transfer function relating the input u and the output y.

File format
-----------

The file should be a csv containing at least three columns with labels
t, u and y in any order.