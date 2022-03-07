#!/bin/bash

make clean; make; ./d2q9-bgk input_$1x$2.params obstacles_$1x$2.dat
# icc -O3 -fast -xCORE-AVX2 d2q9-bgk.c -o d2q9-bgk
# srun --ntasks=1 --cpus-per-task=28 --nodes=1 --exclusive --reservation=COSC026662 --account=COSC026662 --job -name=d2q9 --pty bash
# make clean; rm gmon.out; rm profile.txt; make