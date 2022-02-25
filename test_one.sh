#!/bin/bash

# make clean; rm gmon.out; rm profile.txt; make; ./d2q9-bgk input_$1x$1.params obstacles_$1x$1.dat; gprof d2q9-bgk gmon.out > profile.txt
make clean; rm gmon.out; rm profile.txt; make