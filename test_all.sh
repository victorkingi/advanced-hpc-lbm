#!/bin/bash

make clean; make; ./d2q9-bgk input_128x128.params obstacles_128x128.dat; ./d2q9-bgk input_256x256.params obstacles_256x256.dat; ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat