#!/bin/bash

source /home/blodej/nec_hpc_tools/env/mpi/hpcx-2.9.0/linux-x64-intel2021.3/env.sh

# source ${HOME}/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
# source ${HOME}/nec_hpc_tools/env/compilers/linux-x64-cuda11.2/env.sh

# source ${HOME}/nec_hpc_tools/env/mpi/openmpi-4.1.0/linux-x64-intel2020.2/env.sh



HYPRE_ROOT=${HOME}/HYPRE/hypre_x86

DEBUG=0
ICX=1

PREFIX=${HOME}/HYPRE/build_amd
CC_EXTRAFLAGS=" -march=core-avx2"

DEBUG_FLAGS=" -g"

if [[ ${ICX} -eq 1 ]]
then
    PREFIX=${HOME}/HYPRE/build_icx
    CC_EXTRAFLAGS=" -march=icelake-server"
fi


if [[ ${DEBUG} -eq 1 ]]
then
    FC_FLAGS+=${DEBUG_FLAGS}
    CC_EXTRAFLAGS+=${DEBUG_FLAGS}

fi

rm -rf ${PREFIX}
pushd ${HYPRE_ROOT}/src

make clean

 
CC=mpicc ./configure \
                --enable-shared \
                --disable-fortran \
                --with-MPI \
                --with-openmp \
                --with-extra-CFLAGS="${CC_EXTRAFLAGS}" \
                --with-extra-CXXFLAGS="${CC_EXTRAFLAGS}" \
                --with-blas-libs=mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl \
                --with-blas-lib-dirs="${MKLROOT}/lib/intel64_lin" \
                --with-lapack-libs=mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl \
                --with-lapack-lib-dirs="${MKLROOT}/lib/intel64_lin" \
                --prefix="${PREFIX}"

make -j

make install

popd
