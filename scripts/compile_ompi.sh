#!/bin/bash

# source ${HOME}/nec_hpc_tools/env/compilers/linux-x64-intel2020.2/env.sh
source /home/emorsi/nec_hpc_tools/env/compilers/linux-x64-pgi19.10/env.sh

OPENMPIV=4.1.0

ARCH=linux-x64-pgi19.10
# ARCH=linux-x64-intel2020.2

INSTALL_DIR=$PWD/openmpi-${OPENMPIV}-cuda/${ARCH}

mkdir -p ${INSTALL_DIR}

WORKDIR=/home/emorsi/nec_hpc_tools/OMPI/tmp

mkdir -p ${WORKDIR}

cp openmpi-${OPENMPIV}.tar.gz ${WORKDIR}

cd ${WORKDIR}

tar xvf openmpi-${OPENMPIV}.tar.gz

cd openmpi-${OPENMPIV}

./configure CC=pgcc CXX=pgc++ F77=pgf77 FC=pgfortran --with-hwloc=internal \
    -with-slurm=yes --with-tm=no --enable-mpi1-compatibility \
    --enable-orterun-prefix-by-default \
    --with-cuda=/usr/local/cuda-11 \
    --prefix=${INSTALL_DIR} 2>&1 | tee  ${INSTALL_DIR}/config.${ARCH}-output.log

make -j 24 |& tee  ${INSTALL_DIR}/make.${ARCH}-output.log

make install

rm -rf ${WORKDIR}
