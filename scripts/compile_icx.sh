#!/bin/bash

# source ${HOME}/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
# source ${HOME}/nec_hpc_tools/env/mpi/openmpi-4.1.0/linux-x64-intel2020.2/env.sh
source /home/blodej/nec_hpc_tools/env/mpi/hpcx-2.9.0/linux-x64-intel2021.3/env.sh


mpicc --version 

make -C ../src -f Makefile_icx clean
make -C ../src -f Makefile_icx DEBUG=0


