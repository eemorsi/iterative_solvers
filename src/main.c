
#include "sparse_matrix.h"
#include <stdio.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_struct_mv.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"


int main(int argc, char *argv[])
{
    char *mtx_filepath = argv[1];
    char *mtx_b_filepath = argv[2];

    SparseMatrixCOO tmp_matrix;
    SparseMatrixCSR csr_matrix;
    elem_t *b;

    int myid, num_procs;
    int n, N, pi, pj;
    double h;

    HYPRE_StructGrid grid;
    HYPRE_StructMatrix A;
    HYPRE_StructVector rhs; //b
    HYPRE_StructVector x;

    HYPRE_Solver solver;

    // printf("The total number of rows %d num of col %d\n", csr_matrix.nrows, csr_matrix.ncolumns);
    // split CSR into

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Init();

    /* Figure out the processor grid (N x N).  The local problem size is n^2,
      while pi and pj indicate the position in the processor grid. */
    N = pow(num_procs, 1.0 / 2.0) + 0.5;
    if (num_procs != N * N)
    {
        if (myid == 0)
        {
            printf("Can't run on %d processors, try %d.\n", num_procs, N * N);
        }
        MPI_Finalize();
        exit(1);
    }

    if (myid == 0)
    {
        fast_load_from_mtx_file(mtx_filepath, &tmp_matrix);
        convert_coo_to_csr(&tmp_matrix, &csr_matrix, 1);

        fast_load_from_array_file(mtx_b_filepath, b);
        n = csr_matrix.ncolumns;
    }

    h = 1.0 / (N * n);
    pj = myid / N;
    pi = myid - pj * N;

    {
        int ilower[2] = {1 + pi * n, 1 + pj * n};
        int iupper[2] = {n + pi * n, n + pj * n};

        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector b;
        HYPRE_ParVector par_b;
        HYPRE_IJVector x;
        HYPRE_ParVector par_x;
        
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower[0], ilower[1], iupper[0], iupper[1], &A);
        // Note that this is for a symmetric matrix, ilower/iupper of row and ilower/iupper of column are same
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);
        HYPRE_IJMatrixSetValues(A, local_size, &spdata.nnz_v_[0], &spdata.rows_[0],
                                &spdata.colidx_[0], &spdata.values_[0]);
        HYPRE_IJMatrixAssemble(A);
    }
    /* 1. Set up the grid.  For simplicity we use only one part to represent the
         unit square. */
    // {
    //     int ndim = 2;

    //     /* Create an empty 2D grid object */
    //     HYPRE_StructGridCreate(MPI_COMM_WORLD, ndim, &grid);

    //     /* Set the extents of the grid - each processor sets its grid boxes. */
    //     {
    //         int ilower[2] = {1 + pi * n, 1 + pj * n};
    //         int iupper[2] = {n + pi * n, n + pj * n};
    //         printf("ID %d \tilower(%d,%d)\t iupper(%d,%d)\n", myid, 1 + pi * n, 1 + pj * n, iupper[0], iupper[1]);

    //         HYPRE_StructGridSetExtents(grid, ilower, iupper);
    //     }
    // }

    MPI_Finalize();

    return 0;
}
