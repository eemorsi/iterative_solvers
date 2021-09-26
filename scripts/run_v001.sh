#!/bin/bash

source /opt/nec/ve/nlc/2.1.0/bin/nlcvars.sh
source /opt/nec/ve/mpi/2.11.0/bin/necmpivars.sh

B=0
DEBUG=0
SOL=6
E_MTX=1
LOG=1
P_LOG=1
NTH=8
PADDING=1
NP=1

# MTX=~/data/sherman/5/sherman5.mtx
# MTX_B=~/data/sherman/5/sherman5_rhs1.mtx

# MTX=/home/nec/emorsi/data/orsirr_1.mtx
# MTX=/home/nec/emorsi/data/pores_2.mtx
# MTX=/home/nec/emorsi/sim_data/Bump_2911/Bump_2911.mtx
MTX=/home/nec/emorsi/sim_data/af23560/af23560.mtx

# MTX=/home/nec/emorsi/data/e40r5000/e40r5000.mtx
# MTX_B=/home/nec/emorsi/data/e40r5000/e40r5000_rhs1.mtx



if [[ $PADDING -eq 1 ]]; then
       if [[ $DEBUG -eq 0 ]]; then
              if [[ $B -ne 0 ]]; then
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -emtx ${E_MTX} \
                            -padding \
                            -mtx ${MTX} -b ${MTX_B}
              else
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -padding \
                            -emtx ${E_MTX} \
                            -mtx ${MTX}
              fi
       else
              if [[ $B -ne 0 ]]; then
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 /usr/bin/xterm -hold -e /opt/nec/ve/bin/gdb --args ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -padding \
                            -emtx ${E_MTX} \
                            -mtx ${MTX} -b ${MTX_B}
              else
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 /usr/bin/xterm -hold -e /opt/nec/ve/bin/gdb --args ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -padding \
                            -emtx ${E_MTX} \
                            -mtx ${MTX}
              fi
       fi
else
       if [[ $DEBUG -eq 0 ]]; then
              if [[ $B -ne 0 ]]; then
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -emtx ${E_MTX} \
                            -mtx ${MTX} -b ${MTX_B}
              else
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -emtx ${E_MTX} \
                            -mtx ${MTX}
              fi
       else
              if [[ $B -ne 0 ]]; then
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 /usr/bin/xterm -hold -e /opt/nec/ve/bin/gdb --args ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -emtx ${E_MTX} \
                            -mtx ${MTX} -b ${MTX_B}
              else
                     OMP_NUM_THREADS=${NTH} /opt/nec/ve/bin/mpirun -ve 0 -np 1 /usr/bin/xterm -hold -e /opt/nec/ve/bin/gdb --args ./solver \
                            -solver ${SOL} \
                            -log ${LOG} \
                            -precond_log ${P_LOG} \
                            -niter 1 \
                            -maxit 10000 \
                            -tol 1e-16 \
                            -emtx ${E_MTX} \
                            -mtx ${MTX}
              fi
       fi
fi
