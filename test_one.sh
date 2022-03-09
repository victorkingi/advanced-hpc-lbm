#!/bin/bash

make clean; make; ./d2q9-bgk input_$1x$2.params obstacles_$1x$2.dat
# icc -O3 -fast -xCORE-AVX2 d2q9-bgk.c -o d2q9-bgk
# make clean; rm gmon.out; rm profile.txt; make
