#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sblas.h>

void print(void **arr, int size)
{
    for (size_t i = 0; i < size; i++)
    {
        printf(arr[i]);
    }
}
int main(void)
{
    const int NROWS = 5;
    const int NCOL = 7;
    const int NNZ = 13;
    int i, j;
    /* Values of the nonzero entries of A in row-major order */
    double aval[NNZ] = {
        1.1, 1.2, 2.2, 2.3, 3.3,
        3.4, 3.5, 3.7, 4.1, 4.4,
        5.3, 5.5, 5.6};
    /* Column indices of nonzero entries */
    sblas_int_t iaind[NNZ] = {
        0, 1, 1, 2, 2,
        3, 4, 6, 0, 3,
        2, 4, 5};
    /* Starting points of the rows of the arrays aval and iaind */
    sblas_int_t iaptr[NROWS + 1] = {0, 2, 4, 8, 10, 13};
    /* The vector X */
    double x[NCOL] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    /* The vector Y to be computed */
    double y[NROWS];
    /* Error indicator */
    const int NTHREADS = omp_get_max_threads();
    const int CHUNK = (NROWS / (float)NTHREADS + 0.5);
    fprintf(stderr, "CHUNK %d %d\n", CHUNK, NTHREADS);

#pragma omp parallel private(i, j)
    {
        int iBeg, iEnd;
        // assumption; load per row is constant ..
        int myid = omp_get_thread_num();

        iBeg = CHUNK * myid;
        iEnd = CHUNK * (myid + 1);
        iEnd = iEnd > NROWS ? NROWS : iEnd;
        int l_mrows = iEnd - iBeg;

#if 1
        int l_nnz = iaptr[iEnd] - iaptr[iBeg];
        int *row_idx = (int *)aligned_alloc(128, l_nnz * sizeof(int));
        double *l_y = &y[iBeg];
        for (i = iBeg; i < iEnd; i++)
        {
            for(j=iaptr[i]; j< iaptr[i+1]; j++){
                row_idx[j-iaptr[iBeg]] = i-iBeg;
            }
            l_y[i-iBeg] = 0.0;
        }

        double *l_aval = &aval[iaptr[iBeg]];
        int * l_iaind= &iaind[iaptr[iBeg]];

        for ( i = 0; i < iaptr[iEnd]-iaptr[iBeg]; i++)
        {
            l_y[row_idx[i]] += l_aval[i]*x[l_iaind[i]];
        }
        

#else
        int *row_ptr = (int *)aligned_alloc(128, (l_mrows + 1) * sizeof(int));
        // memcpy(row_ptr, &iaptr[iBeg], l_mrows + 1);
        for (size_t i = iBeg; i <= iEnd; i++)
        {
            row_ptr[i - iBeg] = iaptr[i] - iaptr[iBeg];
        }

        fprintf(stderr, "\n%d\t %d\t%d\n", iBeg, iEnd, l_mrows);
        int ierr;
        /* Set the number of OpenMP threads */
        // omp_set_num_threads(8);
        /* Creation of a handle from CSR format */
        sblas_handle_t a;
        ierr = sblas_create_matrix_handle_from_csr_rd(l_mrows, NCOL, row_ptr, &iaind[iaptr[iBeg]], &aval[iaptr[iBeg]], SBLAS_INDEXING_0, SBLAS_GENERAL, &a);
        /* Analysis of the sparse matrix A */
        ierr = sblas_analyze_mv_rd(SBLAS_NON_TRANSPOSE, a);
        const double alpha = 1.0;
        const double beta = 0.0;
        /* Matrix-vector multiplication Y = A * X */
        // double *l_y = (double *)aligned_alloc(128, l_mrows * sizeof(double));

        ierr = sblas_execute_mv_rd(SBLAS_NON_TRANSPOSE, a, alpha, x, beta, &y[iBeg]);
        // for (size_t i = iBeg; i < iEnd; i++)
        // {
        //     y[i]=l_y[i-iBeg];
        // }

        /* Destruction of the handle */
        ierr = sblas_destroy_matrix_handle(a);
#endif
    }

    /* Print the vector Y */
    printf("%s\n", "******** Vector Y ********");
    for (int i = 0; i < NROWS; i++)
    {
        printf("  %s%1d%s%14.12f\n", "y[", i, "] = ", y[i]);
    }
    printf("%s\n", "********** End ***********");
    return 0;
}
