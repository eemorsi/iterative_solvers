#!/bin/bash

# source ${HOME}/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
# source ${HOME}/nec_hpc_tools/env/compilers/linux-x64-cuda11.2/env.sh


nvhome=/opt/bm/nvidia/hpc_sdk
target=Linux_x86_64
version=20.9

export nvcudadir=$nvhome/$target/$version/cuda
export nvcompdir=$nvhome/$target/$version/compilers
export nvmathdir=$nvhome/$target/$version/math_libs
export nvcommdir=$nvhome/$target/$version/comm_libs

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

export PATH=/usr/local/cuda/cuda-11.2/bin:$PATH

mpicc --version 

SOLVER_ROOT=${HOME}/HYPRE/SpMTXReader
pushd ${SOLVER_ROOT}/src

make -f Makefile_gpu clean
make -f Makefile_gpu DEBUG=0 POWER=1
 
popd

