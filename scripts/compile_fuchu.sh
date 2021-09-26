#!/bin/bash

# make -C src -f Makefile_x86 clean
# make -C src -f Makefile_x86 DEBUG=0


source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
source /opt/nec/ve/mpi/2.16.0/bin/necmpivars.sh

SOLVER_ROOT=/mnt/scatefs_necd/essem/Benchmarks/HYPRE/SpMTXReader
pushd ${SOLVER_ROOT}/src

make clean
make FTRACE=0 DEBUG=0 POWER=1

popd
