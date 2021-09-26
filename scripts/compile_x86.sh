#!/bin/bash

# source ${HOME}/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
source /home/blodej/nec_hpc_tools/env/mpi/hpcx-2.9.0/linux-x64-intel2021.3/env.sh

mpicc --version 

make -C ../src -f Makefile_x86 clean
make -C ../src -f Makefile_x86 DEBUG=0


