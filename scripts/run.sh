#!/bin/bash

source /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
source /opt/nec/ve/mpi/2.16.0/bin/necmpivars.sh

SOLVER=7
DEBUG=0
E_MTX=1

LOG=0
P_LOG=0
NTH=8
PADDING=0
NP=1
VE=0
MAXIT=10000
SOLVER_ROOT=${HOME}/git/HYPRE/SpMTXReader


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
# NAME=raefsky5 RHS=1 X=0 #rhs
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
NAME=RM07R RHS=1 X=1 # contain rhs and sol
# NAME=HV15R RHS=1 X=1 # contain rhs and sol

MTX="/home/nec/emorsi/sim_data/${NAME}/${NAME}.mtx"
MTX_B="/home/nec/emorsi/sim_data/${NAME}/${NAME}_b.mtx"
MTX_X="/home/nec/emorsi/sim_data/${NAME}/${NAME}_x.mtx"


CMD=" ${SOLVER_ROOT}/solver -solver ${SOLVER}  -log ${LOG}  -precond_log ${P_LOG}  -niter 1  -maxit ${MAXIT}  -tol 1e-17  -emtx ${E_MTX} -mtx ${MTX}"

if [[ $PADDING -eq 1 ]]; then
       CMD+="  -padding"
fi
if [[ $RHS -eq 1 ]]; then
       CMD+=" -b ${MTX_B}"
fi

if [[ $X -eq 1 ]]; then
       CMD+=" -x ${MTX_X}"
fi

if [[ $DEBUG -eq 0 ]]; then
       export OMP_NUM_THREADS=${NTH} 
       mpirun -ve ${VE} -np ${NP} ${CMD}
else
       export OMP_NUM_THREADS=${NTH} 
       mpirun -ve ${VE} -np ${NP} /usr/bin/xterm -hold -e /opt/nec/ve/bin/gdb --args ${CMD}
fi
