#include "sparse_matrix.h"

void sort(const SparseMatrixCOO *coo_matrix, SparseMatrixCOO *s_coo_matrix)
{
    int i, j;
    int *freq, *inc, *jmp;

    freq = (int *)aligned_alloc(align_size, coo_matrix->nrows * sizeof(int));
    inc = (int *)aligned_alloc(align_size, coo_matrix->nrows * sizeof(int));
    jmp = (int *)aligned_alloc(align_size, coo_matrix->nrows * sizeof(int));

    printf("the number of rows: %d\n", coo_matrix->nrows);

    // memset(inc, 0, coo_matrix->nrows);
    // memset(freq, 0, coo_matrix->nrows);
    for (i = 0; i < coo_matrix->nrows; i++)
    {
        inc[i] = 0;
        freq[i] = 0;
    }
    for (i = 0; i < coo_matrix->nnz; i++)
    {
        freq[coo_matrix->rows[i]]++;
    }
    // for (i = 0; i < coo_matrix->nrows; i++)
    // {
    //     printf("%d -> %d\n", i, freq[i]);
    // }
    jmp[0] =0;
    for (i = 1; i < coo_matrix->nrows; i++)
    {
        jmp[i]=jmp[i-1]+freq[i-1];
        // freq[i] += freq[i - 1];
        // printf("%d -> %d\n", i, freq[i]);
    }
    s_coo_matrix->values = (double *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(double));
    s_coo_matrix->rows = (int *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(int));
    s_coo_matrix->columns = (int *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(int));

    s_coo_matrix->nnz = coo_matrix->nnz;
    s_coo_matrix->nrows = coo_matrix->nrows;
    s_coo_matrix->ncolumns = coo_matrix->ncolumns;

    for (i = 0; i < coo_matrix->nnz; i++)
    {
        int r = coo_matrix->rows[i];
        int p = jmp[r] + inc[r];
        s_coo_matrix->values[p] = coo_matrix->values[i];
        s_coo_matrix->rows[p] = coo_matrix->rows[i];
        s_coo_matrix->columns[p] = coo_matrix->columns[i];
        inc[r]++;
    }
}
int main(int argc, char const *argv[])
{
    const char *mtx_f = argv[1];

    SparseMatrixCOO tmp, coo;
    fast_load_from_mtx_file(mtx_f, &tmp);
    // for (size_t i = 0; i < tmp.nnz; i++)
    // {
    //     printf("%d %d %f\n", tmp.rows[i], tmp.columns[i], tmp.values[i]);
    // }

    sort(&tmp, &coo);
    // sort_coo_row(&coo);

    for (size_t i = 0; i < coo.nnz; i++)
    {
        printf("%d %d %f\n", coo.rows[i], coo.columns[i], coo.values[i]);
    }

    return 0;
}
