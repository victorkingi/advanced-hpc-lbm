#!/bin/bash

make clean; make; ./d2q9-bgk input_$1x$2.params obstacles_$1x$2.dat
# make clean; rm gmon.out; rm profile.txt; make