/* Copyright 2020 Barcelona Supercomputing Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <stdlib.h>
#include <stdint.h>

// extern unsigned int align_size;

struct SparseMatrixCSR_STRUCT
{
    double *values; // values of matrix entries
    int *column_indices;
    int *row_pointers;
    unsigned int nrows;
    unsigned int ncolumns;
    unsigned int nnz;
};
typedef struct SparseMatrixCSR_STRUCT SparseMatrixCSR;

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

struct SparseMatrixELLPACK_STRUCT
{
    double *values;
    size_t *column_indices;
    size_t max_row_size;
    size_t nrows;
    size_t ncolumns;
    size_t nnz;
};
typedef struct SparseMatrixELLPACK_STRUCT SparseMatrixELLPACK;

// void load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix);
void fast_load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix);
void fast_load_from_array_file(const char *mtx_filepath, double** vals, const int ext_factor);
void convert_coo_to_csr(const SparseMatrixCOO *coo_matrix, SparseMatrixCSR *csr_matrix, int free_coo);
void sort_coo_row(const SparseMatrixCOO *coo_matrix, SparseMatrixCOO *s_coo_matrix);
void sort_coo_row_padding(const SparseMatrixCOO *coo_matrix, SparseMatrixCOO *s_coo_matrix);
void extend_sparse_coo(const SparseMatrixCOO *coo_matrix, SparseMatrixCOO *e_coo_matrix, const int ext_factor, const int free_coo);
void print_full_csr_matrix(const SparseMatrixCSR *csr_matrix);
void extend_array(double *rhs_org, double **rhs, const int size, const int ext);
size_t get_multiple_of_align_size(size_t size);
#endif // SPARSE_MATRIX_H
