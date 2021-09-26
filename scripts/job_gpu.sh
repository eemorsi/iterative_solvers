#!/bin/bash

#source $HOME/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh
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
export LD_LIBRARY_PATH=${HOME}/HYPRE/build_gpu/lib:${LD_LIBRARY_PATH}

export PATH=/usr/local/cuda/cuda-11.2/bin:$PATH


SOLVER=7

DEBUG=0
E_MTX=1

LOG=0
P_LOG=0
PADDING=0
NP=1
NTH=$(expr 36 / $NP)

# NAME=662_bus RHS=0 X=0 
# NAME=windtunnel_evap2d RHS=1 X=0  #rhs
# NAME=sherman3 RHS=0 X=0  #rhs
# NAME=raefsky5 RHS=1 X=0 #rhs -> not solvable with ilu
# NAME=hvdc1 RHS=1 X=0 #hrs
# NAME=hvdc2 RHS=1 X=0 #hrs
# NAME=sherman5 RHS=1 X=0 #rhs
# NAME=airfoil_2d RHS=1 X=0 #rhs
# NAME=orsirr_1 RHS=0 X=0
# NAME=pores_2 RHS=0 X=0
# NAME=bbmat RHS=1 X=1 # contain rhs and sol
# NAME=Bump_2911 RHS=0 X=0
# NAME=Emilia_923 RHS=0 X=0
# NAME=Serena RHS=0 X=0
# NAME=fcondp2 RHS=0 X=0
# NAME=windtunnel_evap3d RHS=1 X=0
# NAME=fullb RHS=0 X=0
# NAME=pwtk RHS=0 X=0
# NAME=StocF-1465 RHS=0 X=0
# NAME=Fault_639 RHS=0 X=0
# NAME=lpi_gosh RHS=0 X=0
# NAME=Geo_1438 RHS=0 X=0
# NAME=ML_Laplace RHS=0 X=0
# NAME=PFlow_742 RHS=0 X=0
# NAME=CoupCons3D RHS=0 X=0
# NAME=PR02R RHS=1 X=1
# NAME=RM07R RHS=1 X=1
# NAME=HV15R RHS=1 X=1
# NAME=scircuit RHS=1 X=0
# NAME=ldoor RHS=0 X=0
# NAME=af_shell10 RHS=0 X=0
# NAME=bundle_adj RHS=1 X=0
# NAME=Flan_1565 RHS=0 X=0;
# NAME=Cube_Coup_dt0 RHS=0 X=0;
# NAME=dielFilterV3real RHS=1 X=0;
# NAME=ML_Geer RHS=0 X=0;

MTX="${HOME}/HYPRE/sim_data/${NAME}/${NAME}.mtx"
MTX_B="${HOME}/HYPRE/sim_data/${NAME}/${NAME}_b.mtx"
MTX_X="${HOME}/HYPRE/sim_data/${NAME}/${NAME}_x.mtx"

DIR="${HOME}/HYPRE/SpMTXReader"

#env

#ulimit -a 

echo "Dataset: ${NAME}"
echo "NThreads: ${NTH}"

echo "CPU INFO:"
lscpu |grep -e "Model name"
lscpu |grep -e "Socket(s):" -e "Core(s) per socket" -e "NUMA node(s):"

CMD="${DIR}/solver_gpu -solver ${SOLVER}  -log ${LOG}  -precond_log ${P_LOG}  -niter 1  -maxit 10000  -tol 1e-16  -emtx ${E_MTX} -mtx ${MTX}"

if [[ $PADDING -eq 1 ]]; then
       CMD+="  -padding"
fi
if [[ $RHS -eq 1 ]]; then
       CMD+=" -b ${MTX_B}"
fi

if [[ $X -eq 1 ]]; then
       CMD+=" -x ${MTX_X}"
fi

#export KMP_AFFINITY=verbose,compact

export KMP_AFFINITY=granularity=fine,compact
export OMP_NUM_THREADS=${NTH}
NVPROF="solver_id.${SOLVER}.${NAME}.proc_id"

#mpirun -np ${NP} --mca btl '^openib'  ${CMD}
mpirun -np ${NP} --mca btl '^openib'  nvprof -f -o ${NVPROF}.%q{OMPI_COMM_WORLD_RANK}.nvprof ${CMD}

exit 0
