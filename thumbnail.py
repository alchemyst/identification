#!/usr/bin/env python

import matplotlib.pyplot as plt
import lvm
import sys
import itertools

def getgroupname(n):
    return n.split('_')[0]

for f in sys.argv[1:]:
    d = lvm.lvm(f)
    plt.figure()
    names = d.data.dtype.names[1:]
    Ngroups = len(set(map(getgroupname, names)))
    namegroups = itertools.groupby(names, getgroupname)
    for i, (groupname, items) in enumerate(namegroups):
        plt.subplot(Ngroups, 1, i)
        plt.ylabel(groupname)
        for name in items:
            normy = d.data[name]
            normy -= min(normy)
            normy /= max(normy)
            t = d.data.X_Value
            tindex = t<400
            plt.plot(t[tindex], normy[tindex], label=name)
        plt.legend(loc='best')

    plt.savefig(f[:-4] + '.png')
