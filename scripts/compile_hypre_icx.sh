#!/bin/bash

source ${HOME}/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
# source ${HOME}/nec_hpc_tools/env/compilers/linux-x64-cuda11.2/env.sh

# source ${HOME}/nec_hpc_tools/env/mpi/openmpi-4.1.0/linux-x64-intel2020.2/env.sh



HYPRE_ROOT=${HOME}/HYPRE/hypre_x86
PREFIX=${HOME}/HYPRE/build_icx

CC_EXTRAFLAGS=" -xhost"
DEBUG_FLAGS=" -g"

DEBUG=0


if [[ ${DEBUG} -eq 1 ]]
then
    FC_FLAGS+=${DEBUG_FLAGS}
    CC_EXTRAFLAGS+=${DEBUG_FLAGS}

fi


pushd ${HYPRE_ROOT}/src

make clean

 
CC=mpicc ./configure \
                --enable-shared \
                --disable-fortran \
                --with-MPI \
                --with-openmp \
                --with-extra-CFLAGS="${CC_EXTRAFLAGS}" \
                --with-extra-CXXFLAGS="${CC_EXTRAFLAGS}" \
                --prefix="${PREFIX}"

make -j

make install

popd