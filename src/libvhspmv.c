#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "mmio.h"
#include <pthread.h>
#include <libvepseudo.h>

#define MAXLEN 500
#define align_size 64

struct SparseMatrixCOO_STRUCT
{
    double *values;  // values of matrix entries
    int *rows;    // row_index
    int *columns; // col_index
    unsigned int nrows;
    unsigned int ncolumns;
    unsigned int nnz;
};
typedef struct SparseMatrixCOO_STRUCT SparseMatrixCOO;


SparseMatrixCOO *coo_matrix;
void fast_load_from_mtx_file(const char *mtx_filepath);


void get_mtx_char(const char *mtx_filepath, unsigned int *nrows, unsigned int *ncolumns, unsigned int *nnz ){
    // printf("[VH] mtx_file : %s\n",mtx_filepath );
    coo_matrix = (SparseMatrixCOO *) malloc(sizeof(SparseMatrixCOO));

    fast_load_from_mtx_file(mtx_filepath);
    // printf("[VH chars] %d \t %d\t %d\n",coo_matrix->nrows, coo_matrix->ncolumns, coo_matrix->nnz );

    *nrows= coo_matrix->nrows;
    *ncolumns= coo_matrix->ncolumns;
    *nnz= coo_matrix->nnz;
}

void get_mtx_data(/*const char *mtx_filepath,*/ double *values, int *rows, int *columns ){
    // coo_matrix = (SparseMatrixCOO *) malloc(sizeof(SparseMatrixCOO));

    // fast_load_from_mtx_file(mtx_filepath);
    // printf("[VH nnz] %d\n", nnz);
    // printf("[VH - 1001th element] %f\t %d\t %d\n",coo_matrix->values[1000],coo_matrix->rows[1000],coo_matrix->columns[1000]);

    memcpy(values,coo_matrix->values, sizeof(double)*coo_matrix->nnz );
    memcpy(rows,coo_matrix->rows, sizeof(int)*coo_matrix->nnz );
    memcpy(columns,coo_matrix->columns, sizeof(int)*coo_matrix->nnz );

}

void fast_load_from_mtx_file(const char *mtx_filepath)
{

    int ret_code;
    unsigned int mtx_rows, mtx_cols, mtx_entries;
    FILE *f;
    MM_typecode matcode;

    if ((f = fopen(mtx_filepath, "r")) == NULL)
    {
        fprintf(stderr, "Could not open file: %s \n", mtx_filepath);
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        exit(1);
    }

    /* 
        This code, will only work with MTX containing: REAL number, Sparse, Matrices.
        Throws an error otherwise. See mmio.h for more information.
    */
    if ((!mm_is_real(matcode) && !mm_is_pattern(matcode)) || !mm_is_matrix(matcode) || !mm_is_sparse(matcode))
    {
        fprintf(stderr, "Market Market type: [%s] not supported\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* Get the number of matrix rows and columns */
    if ((ret_code = mm_read_mtx_crd_size(f, &mtx_rows, &mtx_cols, &mtx_entries)) != 0)
    {
        fprintf(stderr, "Error while reading matrix dimension sizes.\n");
        exit(1);
    }

    long current_stream_position = ftell(f);
    fseek(f, 0, SEEK_END);
    long nnz_string_size = ftell(f) - current_stream_position;
    fseek(f, current_stream_position, SEEK_SET); // Leave the pointer where it was before

    char *nnz_string = (char *)malloc(nnz_string_size + 1);
    fread(nnz_string, 1, nnz_string_size, f);
    fclose(f);

    /* Fill COO struct */
    coo_matrix->nrows = mtx_rows;
    coo_matrix->ncolumns = mtx_cols;

    unsigned int nnz_count = 0;
    if (mm_is_symmetric(matcode))
    {
        unsigned int max_entries = 2 * mtx_entries; // 2 * mtx_entries is an upper bound
        coo_matrix->rows = (int *)malloc(sizeof(int)*max_entries );

        coo_matrix->columns = (int *)malloc(sizeof(int)*max_entries);

        coo_matrix->values = (double *)malloc(sizeof(double)*max_entries);

        // Load Symmetric MTX, note that COO might be unordered.
        if (!mm_is_pattern(matcode))
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (unsigned int i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = strtod(line_ptr, &next_token);
                line_ptr = next_token;

                if (coo_matrix->rows[nnz_count] == coo_matrix->columns[nnz_count])
                {
                    nnz_count++;
                }
                else
                {
                    coo_matrix->rows[nnz_count + 1] = coo_matrix->columns[nnz_count];
                    coo_matrix->columns[nnz_count + 1] = coo_matrix->rows[nnz_count];
                    coo_matrix->values[nnz_count + 1] = coo_matrix->values[nnz_count];
                    nnz_count = nnz_count + 2;
                }
            }
        }
        else
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (unsigned int i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = 1.0f;
                // fprintf(stderr, " %lu %lu %lf\n", coo_matrix->rows[nnz_count], coo_matrix->columns[nnz_count], coo_matrix->values[nnz_count]);
                if (coo_matrix->rows[nnz_count] == coo_matrix->columns[nnz_count])
                {
                    nnz_count++;
                }
                else
                {
                    coo_matrix->rows[nnz_count + 1] = coo_matrix->columns[nnz_count];
                    coo_matrix->columns[nnz_count + 1] = coo_matrix->rows[nnz_count];
                    coo_matrix->values[nnz_count + 1] = 1.0f;
                    nnz_count = nnz_count + 2;
                }
            }
        }
    }
    else
    {
        coo_matrix->rows = (int *)malloc(sizeof(int)*mtx_entries);

        coo_matrix->columns = (int *)malloc(sizeof(int)*mtx_entries);

        coo_matrix->values = (double *)malloc(sizeof(double)*mtx_entries);

        if (!mm_is_pattern(matcode))
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (unsigned int i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = strtod(line_ptr, &next_token);
                line_ptr = next_token;
                nnz_count++;
            }
        }
        else
        {
            char *line_ptr = nnz_string;
            char *next_token;

            for (unsigned int i = 0; i < mtx_entries; i++)
            {
                coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
                line_ptr = next_token;
                coo_matrix->values[nnz_count] = 1.0f;
                nnz_count++;
            }
        }
    }
    // TODO: REMOVE EXPLICIT 0's. apparently some matrices have few (~0.3%). it does not affect the GFLOPS per se.
    coo_matrix->nnz = nnz_count;
        // printf("[VH - 2] %d \t %d\t %d\n",coo_matrix->nrows, coo_matrix->ncolumns, coo_matrix->nnz );

    free(nnz_string);
}

int main(void){

const char* mtx_f = "sherman1.mtx";
unsigned int nrows,  ncolumns,  nnz ;
get_mtx_char(mtx_f,  &nrows,  &ncolumns,  &nnz );
}