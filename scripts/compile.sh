#!/bin/bash

# make -C src -f Makefile_x86 clean
# make -C src -f Makefile_x86 DEBUG=0


source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
source /opt/nec/ve/mpi/2.16.0/bin/necmpivars.sh

export NMPI_F90=/opt/nec/ve/nfort/3.2.0/bin/nfort
export NMPI_CC=/opt/nec/ve/ncc/3.2.0/bin/ncc
export NMPI_CXX=/opt/nec/ve/ncc/3.2.0/bin/nc++

SOLVER_ROOT=${HOME}/git/HYPRE/iterative_solvers
pushd ${SOLVER_ROOT}/src

make clean
make FTRACE=0 DEBUG=0 POWER=1

popd
