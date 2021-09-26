#!/bin/bash

source $HOME/nec_hpc_tools/env/mpi/hpcx-2.7.0/linux-x64-intel2018.3/env.sh

SOLVER=12

E_MTX=1
LOG=0
P_LOG=0
NTH=64
PADDING=1
NP=1
VE="0"

# MTX=~/data/sherman/5/sherman5.mtx
# MTX_B=~/data/sherman/5/sherman5_rhs1.mtx

# MTX=/home/nec/emorsi/data/orsirr_1.mtx
# MTX=/home/nec/emorsi/data/pores_2.mtx
# MTX=/home/nec/emorsi/sim_data/Bump_2911/Bump_2911.mtx
# MTX=/home/nec/emorsi/sim_data/af23560/af23560.mtx

# MTX=/home/nec/emorsi/data/e40r5000/e40r5000.mtx
# MTX_B=/home/nec/emorsi/data/e40r5000/e40r5000_rhs1.mtx

# NAME=662_bus RHS=0 X=0 
# NAME=windtunnel_evap2d RHS=1 X=0  #rhs
# NAME=sherman3 RHS=1 X=0  #rhs
# NAME=sherman5 RHS=1 X=0 #rhs
# NAME=raefsky5 RHS=1 X=0 #rhs -> not solvable with ilu
# NAME=hvdc2 RHS=1 X=0
# NAME=hvdc1 RHS=1 X=0
# NAME=airfoil_2d RHS=1 X=0
#NAME=orsirr_1 RHS=0 X=0
# NAME=pores_2 RHS=0 X=0
NAME=bbmat RHS=1 X=1  # contain rhs and sol

MTX="${HOME}/HYPRE/sim_data/${NAME}/${NAME}.mtx"
MTX_B="${HOME}/HYPRE/sim_data/${NAME}/${NAME}_b.mtx"
MTX_X="${HOME}/HYPRE/sim_data/${NAME}/${NAME}_x.mtx"

DIR="${HOME}/HYPRE/SpMTXReader"

CMD="${DIR}/solver_x86 -solver ${SOLVER}  -log ${LOG}  -precond_log ${P_LOG}  -niter 1  -maxit 10000  -tol 1e-16  -emtx ${E_MTX} -mtx ${MTX}"

if [[ $PADDING -eq 1 ]]; then
       CMD+="  -padding"
fi
if [[ $RHS -eq 1 ]]; then
       CMD+=" -b ${MTX_B}"
fi

if [[ $X -eq 1 ]]; then
       CMD+=" -x ${MTX_X}"
fi

export OMP_NUM_THREADS=${NTH} 

mpirun -np ${NP} ${CMD}
