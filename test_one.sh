#!/bin/bash

make clean; make; ./d2q9-bgk input_$1x$1.params obstacles_$1x$1.dat
# make clean; rm gmon.out; rm profile.txt; make