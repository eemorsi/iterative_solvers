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

export CPATH=$nvcudadir/include
export CPATH=$nvmathdir/include:${CPATH}
export CPATH=${OPAL_PREFIX}/include:${CPATH}
export CPATH=$nvcommdir/nccl/include:${CPATH}
export CPATH=$nvcommdir/nvshmem/include:${CPATH}

export MANPATH=$nvcompdir/man


This is OpenMPI 3.5, I run it with:
mpirun --mca btl '^openib' --hostfile ${HOSTFILE} -np ${NP}

