#! /usr/bin/python

N = 64

import sys
import numpy

r_seed       = float(sys.argv[1])
numpy.random.seed([r_seed])

f  = open('initial_data.dat' , 'w')
f.write("".join(numpy.ndarray.flatten(numpy.array([[[ "%s %s %s %1.3f %1.3f\n" % (i, j, k, numpy.random.normal(0, 0.3), numpy.random.normal(0, 0.3)) for k in range(N)] for j in range(N)] for i in range(N)]))))
f.close()
