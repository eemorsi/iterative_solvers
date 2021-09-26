#!/bin/bash

# source ${HOME}/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
# source ${HOME}/nec_hpc_tools/env/compilers/linux-x64-cuda11.2/env.sh

# source /home/emorsi/nec_hpc_tools/env/mpi/openmpi-4.1.0/linux-x64-intel2020.2/env.sh

nvhome=/opt/bm/nvidia/hpc_sdk
target=Linux_x86_64
version=20.9

nvcudadir=$nvhome/$target/$version/cuda
nvcompdir=$nvhome/$target/$version/compilers
nvmathdir=$nvhome/$target/$version/math_libs
nvcommdir=$nvhome/$target/$version/comm_libs

export NVHPC=$nvhome
export CC=$nvcompdir/bin/nvc
export CXX=$nvcompdir/bin/nvc++
export FC=$nvcompdir/bin/nvfortran
export F90=$nvcompdir/bin/nvfortran
export F77=$nvcompdir/bin/nvfortran
export CPP=cpp

export OPAL_PREFIX=$nvcommdir/mpi

export PATH=$nvcudadir/bin:${PATH}
export PATH=$nvcompdir/bin:${PATH}
export PATH=${OPAL_PREFIX}/bin:${PATH}

export LD_LIBRARY_PATH=$nvcudadir/lib64
export LD_LIBRARY_PATH=$nvcompdir/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$nvmathdir/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${OPAL_PREFIX}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$nvcommdir/nccl/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$nvcommdir/nvshmem/lib:${LD_LIBRARY_PATH}


HYPRE_ROOT=${HOME}/HYPRE/hypre_x86
PREFIX=${HOME}/HYPRE/build_gpu
rm -rf ${PREFIX} >/dev/null

# CC_EXTRAFLAGS=" -fopenmp -O2 -march=core-avx2"
CC_EXTRAFLAGS=" "
DEBUG_FLAGS=" -g"

DEBUG=0


if [[ ${DEBUG} -eq 1 ]]
then
    FC_FLAGS+=${DEBUG_FLAGS}
    CC_EXTRAFLAGS+=${DEBUG_FLAGS}

fi

# mpicc --version


pushd ${HYPRE_ROOT}/src
CUDA_HOME=/usr/local/cuda-11.2
make clean
# HYPRE_CUDA_SM='60 70 75 80'
export PATH=/usr/local/cuda-11.2/bin:$PATH
CC=mpicc CUDA_HOME=/usr/local/cuda-11.2 HYPRE_CUDA_SM='70' ./configure \
                --enable-shared \
                --disable-fortran \
                --with-MPI \
                --with-MPI-include=${nvcommdir}/mpi/include \
                --with-MPI-lib-dirs=${nvcommdir}/mpi/lib \
                --with-MPI-libs="nsl mpi" \
                --with-openmp \
                --with-extra-CFLAGS="${CC_EXTRAFLAGS}" \
                --with-extra-CXXFLAGS="${CC_EXTRAFLAGS}" \
                --enable-unified-memory \
                --with-cuda \
                --enable-curand \
                --enable-cublas \
                --enable-cuda-streams \
                --prefix="${PREFIX}"


make -j

make install

popd