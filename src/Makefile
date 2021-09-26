# mpinfort  -Dalign_size=128 -fopenmp -O3 -I/home/nec/emorsi/git/HYPRE/build_nftrace/include ex5big.c mmio.c mytimer.c utils.c sparse_matrix.c -o solver -L/home/nec/emorsi/git/HYPRE/build_nftrace/lib -lHYPRE -L/opt/nec/ve/nlc/2.1.0/lib/ -lblas_openmp -lsblas_openmp -L/opt/nec/ve/nlc/2.1.0/lib/ -llapack

CC          = mpincc
GCC			= gcc
CFLAGS      = -O3 -Wall -fopenmp -report-all -fdiag-vector=3 -fcse-after-vectorization  -msched-interblock -Dalign_size=128

HYPRE_ROOT	= /home/nec/emorsi/git/HYPRE/build
HYPRE_F_ROOT	= /home/nec/emorsi/git/HYPRE/build_ftrace
FTRACE	?=0
POWER	?=0

ifeq ($(POWER), 1)
CFLAGS      += -D__POWER=1
endif
ifeq ($(FTRACE), 0)
	INC			= -I${HYPRE_ROOT}/include
	HYPRE		= -L${HYPRE_ROOT}/lib -lHYPRE 
else
	INC			= -I${HYPRE_F_ROOT}/include
	HYPRE		= -L${HYPRE_F_ROOT}/lib -lHYPRE 
	CFLAGS		+= -ftrace
endif

DEBUG	?=0
ifeq ($(DEBUG), 1)
CFLAGS	+=	-g -traceback=verbose
endif

BLAS		= -L/opt/nec/ve/nlc/2.3.0/lib/ -lblas_openmp -lsblas_openmp 
# BLAS		= -L/opt/nec/ve/nlc/2.1.0/lib/ -lblas_sequential -lsblas_sequential 
# LAPACK		= -L/opt/nec/ve/nlc/2.1.0/lib/ -llapack
LIBS		= ${HYPRE} ${BLAS} ${LAPACK} -lasl_openmp

COM_SRCS := mmio.c sparse_matrix.c
MPI_SRCS := solvers.c ${COM_SRCS}
OMP_SRCS := solvers_omp.c ${COM_SRCS}
MPI_OBJS := $(MPI_SRCS:%.c=%.o)
OMP_OBJS := $(OMP_SRCS:%.c=%.o)

BUILD_DIR	= $(shell pwd)/..

.PHONY = all clean
all: solver libvhspmv.so

solver_omp:	$(OMP_OBJS)
	${CC} ${CFLAGS} ${INC} $^ -o $@ ${LIBS}

solver:	$(MPI_OBJS)
	${CC} ${CFLAGS} ${INC} $^ -o ${BUILD_DIR}/$@ ${LIBS}
libvhspmv.so: mmio.c libvhspmv.c 
	$(GCC) --std=c99 $^ -o ${BUILD_DIR}/$@ -fpic -shared -I/opt/nec/ve/veos/include

%.o: %.c
	${CC} ${CFLAGS} ${INC} -c $<

clean:
	rm -rvf *.o ${BUILD_DIR}/solver ${BUILD_DIR}/solver_omp
	
