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

#include "sparse_matrix.h"
#include "macros.h"
#include "mmio.h"
#include <inttypes.h>
#include <libgen.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __ve__
#include <asl.h>
#include <libvhcall.h>
#define MAXLEN 500
#endif

void fast_load_from_array_file(const char *mtx_filepath, double **vals,
                               const int ext_factor) {

  int ret_code;
  unsigned int m, n;

  FILE *f;
  MM_typecode matcode;

  if ((f = fopen(mtx_filepath, "r")) == NULL) {
    fprintf(stderr, "Could not open file: %s \n", mtx_filepath);
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    fprintf(stderr, "Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*
      This code, will only work with MTX containing: REAL number, Sparse,
     Matrices. Throws an error otherwise. See mmio.h for more information.
  */
  if (!(mm_is_real(matcode)) && !(mm_is_array(matcode))) {
    fprintf(stderr, "Market Market type: [%s] not array\n",
            mm_typecode_to_str(matcode));
    exit(1);
  }

  if (ret_code = mm_read_mtx_array_size(f, &m, &n) != 0) {
    fprintf(stderr, "Error while reading matrix dimension sizes.\n");
    exit(1);
  }

  *vals = (double *)aligned_alloc(align_size, m * ext_factor * sizeof(double));

  printf("Number of allocated elements %d\n", m);
  double tmp;
  for (unsigned int i = 0; i < m; i++) {
    fscanf(f, "%lg\n", &tmp);
    for (int j = 0; j < ext_factor; j++)
      (*vals)[i * ext_factor + j] = tmp;
    // printf("%f\n", vals[i]);
  }
  // char *nnz_string = (char *)malloc(m + 1);
  // fread(nnz_string, 1, m, f);
  // fclose(f);
  // // printf("number of raw %lg\n", vals[10]);
  // char *line_ptr = nnz_string;
  // char *next_token;
  // for (unsigned int i = 0; i < m; i++)
  // {
  //     vals[i] = strtod(line_ptr, &next_token);
  //     line_ptr = next_token;
  //     printf("%lg\n", vals[i]);
  // }
}

void extend_array(double *rhs_org, double **rhs, const int size,
                  const int ext) {
  int i;
  *rhs = (double *)aligned_alloc(align_size, size * ext * sizeof(double));
  // for (i = 0; i < size; i++)
  // {
  //     for (j = 0; j < ext; j++)
  //     {

  //         (*rhs)[i * ext + j] = rhs_org[i];
  //     }
  // }

  for (i = 0; i < size * ext; i++) {

    (*rhs)[i] = rhs_org[i / ext];
  }
}
void get_mtx_size(const char *mtx_filepath, SparseMatrixCOO *coo_matrix) {

  int ret_code;
  unsigned int mtx_rows, mtx_cols, mtx_entries;
  FILE *f;
  MM_typecode matcode;

  if ((f = fopen(mtx_filepath, "r")) == NULL) {
    fprintf(stderr, "Could not open file: %s \n", mtx_filepath);
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    fprintf(stderr, "Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*
      This code, will only work with MTX containing: REAL number, Sparse,
     Matrices. Throws an error otherwise. See mmio.h for more information.
  */
  if ((!mm_is_real(matcode) && !mm_is_pattern(matcode)) ||
      !mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
    fprintf(stderr, "Market Market type: [%s] not supported\n",
            mm_typecode_to_str(matcode));
    exit(1);
  }

  /* Get the number of matrix rows and columns */
  if ((ret_code =
           mm_read_mtx_crd_size(f, &mtx_rows, &mtx_cols, &mtx_entries)) != 0) {
    fprintf(stderr, "Error while reading matrix dimension sizes.\n");
    exit(1);
  }
}

/*
    Loads a Sparse Matrix from a .MTX file format
    into a SparseMatrixCOO data structure.

    More info about the MTX format @ https://math.nist.gov/MatrixMarket/
    Find the biggest MTX repository @ https://sparse.tamu.edu/
*/
#ifdef __ve__
void fast_load_from_mtx_file(const char *mtx_filepath,
                             SparseMatrixCOO *coo_matrix) {

  char buf[MAXLEN];
  int64_t symid;
  vhcall_handle handle;
  vhcall_args *ca;
  uint64_t retval = -1, ret;
  struct timespec t_strt, t_end;

  /* Load VH C library */
//   printf("Load VH C library\n");
  handle = vhcall_install("./libvhspmv.so");
  if (handle == (vhcall_handle)-1) {
    perror("vhcall_install");
    exit(0);
    // goto ret;
  }

  //   const char *mtx_filepath = "RM07R.mtx";
  //   SparseMatrixCOO coo_matrix;
  symid = vhcall_find(handle, "get_mtx_char");
  ca = vhcall_args_alloc();

  memcpy(buf, mtx_filepath, strlen(mtx_filepath));
  memset(buf + strlen(mtx_filepath), '\0', MAXLEN - strlen(mtx_filepath));
  // 1st;
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_IN, 0, buf, MAXLEN);
  // 2nd
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_OUT, 1, &coo_matrix->nrows,
                                sizeof(coo_matrix->nrows));
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_OUT, 2, &coo_matrix->ncolumns,
                                sizeof(coo_matrix->ncolumns));
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_OUT, 3, &coo_matrix->nnz,
                                sizeof(coo_matrix->nnz));
  ret = vhcall_invoke_with_args(symid, ca, &retval);

//   printf("[VE characteristics] %d \t %d\t %d\n", coo_matrix->nrows,
//          coo_matrix->ncolumns, coo_matrix->nnz);

  coo_matrix->values = (double *)malloc(sizeof(double) * coo_matrix->nnz);
  coo_matrix->rows = (int *)malloc(sizeof(int) * coo_matrix->nnz);
  coo_matrix->columns = (int *)malloc(sizeof(int) * coo_matrix->nnz);

  vhcall_args_free(ca);

  symid = vhcall_find(handle, "get_mtx_data");
  ca = vhcall_args_alloc();
  // ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_IN, 0, buf, MAXLEN);
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_OUT, 0, coo_matrix->values,
                                sizeof(double) * coo_matrix->nnz);
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_OUT, 1, coo_matrix->rows,
                                sizeof(int) * coo_matrix->nnz);
  ret = vhcall_args_set_pointer(ca, VHCALL_INTENT_OUT, 2, coo_matrix->columns,
                                sizeof(int) * coo_matrix->nnz);
  // ret = vhcall_args_set_u32(ca, 3,coo_matrix.nnz);
  clock_gettime(CLOCK_REALTIME, &t_strt);
  ret = vhcall_invoke_with_args(symid, ca, &retval);
  clock_gettime(CLOCK_REALTIME, &t_end);

  double time = (t_end.tv_sec - t_strt.tv_sec) * 1e3 +
                (t_end.tv_nsec - t_strt.tv_nsec) * 1e-6;

  fprintf(stderr, "Dataset reading time: %lg msec\n", time);

//   printf("[VE - 1001th element] %f\t %d\t %d\n", coo_matrix->values[1000],
        //  coo_matrix->rows[1000], coo_matrix->columns[1000]);
  vhcall_args_free(ca);

  if (vhcall_uninstall(handle))
    perror("vhcall_uninstall");
}

#else
void fast_load_from_mtx_file(const char *mtx_filepath,
                             SparseMatrixCOO *coo_matrix) {

  int ret_code;
  unsigned int mtx_rows, mtx_cols, mtx_entries;
  FILE *f;
  MM_typecode matcode;

  if ((f = fopen(mtx_filepath, "r")) == NULL) {
    fprintf(stderr, "Could not open file: %s \n", mtx_filepath);
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    fprintf(stderr, "Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*
      This code, will only work with MTX containing: REAL number, Sparse,
     Matrices. Throws an error otherwise. See mmio.h for more information.
  */
  if ((!mm_is_real(matcode) && !mm_is_pattern(matcode)) ||
      !mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
    fprintf(stderr, "Market Market type: [%s] not supported\n",
            mm_typecode_to_str(matcode));
    exit(1);
  }

  /* Get the number of matrix rows and columns */
  if ((ret_code =
           mm_read_mtx_crd_size(f, &mtx_rows, &mtx_cols, &mtx_entries)) != 0) {
    fprintf(stderr, "Error while reading matrix dimension sizes.\n");
    exit(1);
  }

  long current_stream_position = ftell(f);
  fseek(f, 0, SEEK_END);
  long nnz_string_size = ftell(f) - current_stream_position;
  fseek(f, current_stream_position,
        SEEK_SET); // Leave the pointer where it was before

  char *nnz_string = (char *)malloc(nnz_string_size + 1);
  fread(nnz_string, 1, nnz_string_size, f);
  fclose(f);

  /* Fill COO struct */
  coo_matrix->nrows = mtx_rows;
  coo_matrix->ncolumns = mtx_cols;

  unsigned int nnz_count = 0;
  if (mm_is_symmetric(matcode)) {
    unsigned int max_entries =
        2 * mtx_entries; // 2 * mtx_entries is an upper bound
    coo_matrix->rows =
        (int *)aligned_alloc(align_size, max_entries * sizeof(int));

    coo_matrix->columns =
        (int *)aligned_alloc(align_size, max_entries * sizeof(int));

    coo_matrix->values =
        (double *)aligned_alloc(align_size, max_entries * sizeof(double));

    // Load Symmetric MTX, note that COO might be unordered.
    if (!mm_is_pattern(matcode)) {
      char *line_ptr = nnz_string;
      char *next_token;

      for (unsigned int i = 0; i < mtx_entries; i++) {
        coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->values[nnz_count] = strtod(line_ptr, &next_token);
        line_ptr = next_token;

        if (coo_matrix->rows[nnz_count] == coo_matrix->columns[nnz_count]) {
          nnz_count++;
        } else {
          coo_matrix->rows[nnz_count + 1] = coo_matrix->columns[nnz_count];
          coo_matrix->columns[nnz_count + 1] = coo_matrix->rows[nnz_count];
          coo_matrix->values[nnz_count + 1] = coo_matrix->values[nnz_count];
          nnz_count = nnz_count + 2;
        }
      }
    } else {
      char *line_ptr = nnz_string;
      char *next_token;

      for (unsigned int i = 0; i < mtx_entries; i++) {
        coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->values[nnz_count] = 1.0f;
        // fprintf(stderr, " %lu %lu %lf\n", coo_matrix->rows[nnz_count],
        // coo_matrix->columns[nnz_count], coo_matrix->values[nnz_count]);
        if (coo_matrix->rows[nnz_count] == coo_matrix->columns[nnz_count]) {
          nnz_count++;
        } else {
          coo_matrix->rows[nnz_count + 1] = coo_matrix->columns[nnz_count];
          coo_matrix->columns[nnz_count + 1] = coo_matrix->rows[nnz_count];
          coo_matrix->values[nnz_count + 1] = 1.0f;
          nnz_count = nnz_count + 2;
        }
      }
    }
  } else {
    coo_matrix->rows =
        (int *)aligned_alloc(align_size, mtx_entries * sizeof(int));

    coo_matrix->columns =
        (int *)aligned_alloc(align_size, mtx_entries * sizeof(int));

    coo_matrix->values =
        (double *)aligned_alloc(align_size, mtx_entries * sizeof(double));

    if (!mm_is_pattern(matcode)) {
      char *line_ptr = nnz_string;
      char *next_token;

      for (unsigned int i = 0; i < mtx_entries; i++) {
        coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->values[nnz_count] = strtod(line_ptr, &next_token);
        line_ptr = next_token;
        nnz_count++;
      }
    } else {
      char *line_ptr = nnz_string;
      char *next_token;

      for (unsigned int i = 0; i < mtx_entries; i++) {
        coo_matrix->rows[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->columns[nnz_count] = strtoul(line_ptr, &next_token, 10) - 1;
        line_ptr = next_token;
        coo_matrix->values[nnz_count] = 1.0f;
        nnz_count++;
      }
    }
  }
  // TODO: REMOVE EXPLICIT 0's. apparently some matrices have few (~0.3%). it
  // does not affect the GFLOPS per se.
  coo_matrix->nnz = nnz_count;
  free(nnz_string);
}
#endif

void extend_sparse_coo(const SparseMatrixCOO *coo_matrix,
                       SparseMatrixCOO *e_coo_matrix, const int ext_factor,
                       const int free_coo) {
  int ii, i, j;

  const int FACTOR = ext_factor * ext_factor;

  e_coo_matrix->values = (double *)aligned_alloc(
      align_size, coo_matrix->nnz * FACTOR * sizeof(double));
  e_coo_matrix->rows =
      (int *)aligned_alloc(align_size, coo_matrix->nnz * FACTOR * sizeof(int));
  e_coo_matrix->columns =
      (int *)aligned_alloc(align_size, coo_matrix->nnz * FACTOR * sizeof(int));

  e_coo_matrix->nnz = coo_matrix->nnz * FACTOR;
  e_coo_matrix->nrows = coo_matrix->nrows * ext_factor;
  e_coo_matrix->ncolumns = coo_matrix->ncolumns * ext_factor;

  for (ii = 0; ii < coo_matrix->nnz; ii++) {
    for (i = 0; i < ext_factor; i++) {
      for (j = 0; j < ext_factor; j++) {

        e_coo_matrix->values[ii * FACTOR + i * ext_factor + j] =
            coo_matrix->values[ii];
        e_coo_matrix->rows[ii * FACTOR + i * ext_factor + j] =
            coo_matrix->rows[ii] * ext_factor + i;
        e_coo_matrix->columns[ii * FACTOR + i * ext_factor + j] =
            coo_matrix->columns[ii] * ext_factor + j;
      }
    }
  }

  if (free_coo) {
    free(coo_matrix->values);
    free(coo_matrix->rows);
    free(coo_matrix->columns);
  }

  fprintf(stderr, "Finish extending data ..\n");
}
void sort_coo_row(const SparseMatrixCOO *coo_matrix,
                  SparseMatrixCOO *s_coo_matrix) {
  int i;

#ifdef __ve__

  asl_sort_t sort;
  asl_library_initialize();
  asl_sort_create_i32(&sort, ASL_SORTORDER_ASCENDING,
                      ASL_SORTALGORITHM_AUTO_STABLE);
  asl_sort_preallocate(sort, coo_matrix->nnz);

  int *idx = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  int *vals_idx =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);

  s_coo_matrix->values =
      (double *)aligned_alloc(align_size, sizeof(double) * coo_matrix->nnz);
  s_coo_matrix->rows =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);

  s_coo_matrix->columns =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  //   int *tcolumns =
  //       (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  s_coo_matrix->nnz = coo_matrix->nnz;
  s_coo_matrix->nrows = coo_matrix->nrows;
  s_coo_matrix->ncolumns = coo_matrix->ncolumns;

  for (i = 0; i < coo_matrix->nnz; i++) {
    vals_idx[i] = i;
  }
  //   asl_sort_execute_i32(sort, coo_matrix->nnz, s_coo_matrix->rows,
  //                        ASL_NULL, s_coo_matrix->rows,
  //                        idx);

  asl_sort_execute_i32(sort, coo_matrix->nnz, coo_matrix->rows, vals_idx,
                       s_coo_matrix->rows, idx);

  for (i = 0; i < coo_matrix->nnz; i++) {
    s_coo_matrix->values[i] = coo_matrix->values[idx[i]];
    s_coo_matrix->columns[i] = coo_matrix->columns[idx[i]];
  }
  free(idx);
  free(vals_idx);

  /* Sorting Finalization */
  asl_sort_destroy(sort);
  /* Library Finalization */
  asl_library_finalize();

#else
  int *freq, *inc, *jmp;

  freq = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nrows);
  inc = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nrows);
  jmp = (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nrows);

  // printf("the number of rows: %d\n", coo_matrix->nrows);

  // memset(inc, 0, coo_matrix->nrows);
  // memset(freq, 0, coo_matrix->nrows);
  for (i = 0; i < coo_matrix->nrows; i++) {
    inc[i] = 0;
    freq[i] = 0;
  }
  for (i = 0; i < coo_matrix->nnz; i++) {
    freq[coo_matrix->rows[i]]++;
  }

  jmp[0] = 0;
  for (i = 1; i < coo_matrix->nrows; i++) {
    jmp[i] = jmp[i - 1] + freq[i - 1];
    // freq[i] += freq[i - 1];
    // printf("%d -> %d\n", i, freq[i]);
  }
  s_coo_matrix->values =
      (double *)aligned_alloc(align_size, sizeof(double) * coo_matrix->nnz);
  s_coo_matrix->rows =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);
  s_coo_matrix->columns =
      (int *)aligned_alloc(align_size, sizeof(int) * coo_matrix->nnz);

  s_coo_matrix->nnz = coo_matrix->nnz;
  s_coo_matrix->nrows = coo_matrix->nrows;
  s_coo_matrix->ncolumns = coo_matrix->ncolumns;

  for (i = 0; i < coo_matrix->nnz; i++) {
    int r = coo_matrix->rows[i];
    int p = jmp[r] + inc[r];
    s_coo_matrix->values[p] = coo_matrix->values[i];
    s_coo_matrix->rows[p] = coo_matrix->rows[i];
    s_coo_matrix->columns[p] = coo_matrix->columns[i];
    inc[r]++;
  }

  //   printf("compare rows %d\n", coo_matrix->nnz);
  //   for (i = 0; i < coo_matrix->nnz; i++) {
  //     if (s_coo_matrix->columns[i] != ts_coo_matrix.columns[i])
  //       printf("col: %d\t%d\n", s_coo_matrix->columns[i],
  //       ts_coo_matrix.columns[i]);
  //     if (s_coo_matrix->rows[i] != ts_coo_matrix.rows[i])
  //       printf("row: %d\t%d\n", s_coo_matrix->rows[i],
  //       ts_coo_matrix.rows[i]);
  //     if (s_coo_matrix->values[i] != ts_coo_matrix.values[i])
  //       printf("val: %f\t%f\n", s_coo_matrix->values[i],
  //       ts_coo_matrix.values[i]);
  //   }
  //   exit(0);
#endif
}


void sort_coo_row_padding(const SparseMatrixCOO *coo_matrix,
                          SparseMatrixCOO *s_coo_matrix) {
  unsigned int i, j, jj, max_jmp, nelems;
  int *freq, *inc, *jmp;

  freq = (int *)aligned_alloc(align_size, coo_matrix->nrows * sizeof(int));
  inc = (int *)aligned_alloc(align_size, coo_matrix->nrows * sizeof(int));
  jmp = (int *)aligned_alloc(align_size, coo_matrix->nrows * sizeof(int));

  // printf("the number of rows: %d\n", coo_matrix->nrows);

  // memset(inc, 0, coo_matrix->nrows);
  // memset(freq, 0, coo_matrix->nrows);
  for (i = 0; i < coo_matrix->nrows; i++) {
    inc[i] = 0;
    freq[i] = 0;
  }
  for (i = 0; i < coo_matrix->nnz; i++) {
    freq[coo_matrix->rows[i]]++;
  }

  max_jmp = freq[0];
  for (i = 1; i < coo_matrix->nrows; i++) {
    max_jmp = max_jmp < freq[i] ? freq[i] : max_jmp;
  }
  // max_jmp = get_multiple_of_align_size(max_jmp);

  nelems = coo_matrix->nrows * max_jmp;
  // printf("padded row %d", max_jmp);

  // jmp[0] = 0;
  // for (i = 1; i < coo_matrix->nrows; i++)
  // {
  //     jmp[i] = jmp[i - 1] + freq[i - 1];
  //     // freq[i] += freq[i - 1];
  //     // printf("%d -> %d\n", i, freq[i]);
  // }
  s_coo_matrix->values =
      (double *)aligned_alloc(align_size, nelems * sizeof(double));
  s_coo_matrix->rows = (int *)aligned_alloc(align_size, nelems * sizeof(int));
  s_coo_matrix->columns =
      (int *)aligned_alloc(align_size, nelems * sizeof(int));
  // printf("padded row %d", max_jmp);

  /**/

  // for (i = 0; i < coo_matrix->nrows; i++)
  // {
  //     for (j = 0; j < max_jmp; j++)
  //     {
  //         s_coo_matrix->rows[i * max_jmp + j] = i;
  //         s_coo_matrix->values[i * max_jmp + j] = 0.0;
  //         // think about a better way!!!
  //         s_coo_matrix->columns[i * max_jmp + j] = coo_matrix->ncolumns - 1 -
  //         i;
  //     }
  // }

  for (i = 0; i < nelems; i++) {
    s_coo_matrix->rows[i] = i / max_jmp;
    s_coo_matrix->values[i] = 0.0;
    // fill cols with random values
    s_coo_matrix->columns[i] = -1;
  }

  s_coo_matrix->nnz = nelems;
  s_coo_matrix->nrows = coo_matrix->nrows;
  s_coo_matrix->ncolumns = coo_matrix->ncolumns;

  for (i = 0; i < coo_matrix->nnz; i++) {
    int r = coo_matrix->rows[i];
    int p = r * max_jmp + inc[r];
    s_coo_matrix->values[p] = coo_matrix->values[i];
    s_coo_matrix->rows[p] = coo_matrix->rows[i];
    s_coo_matrix->columns[p] = coo_matrix->columns[i];

    inc[r]++;
  }

/* assume col order also ..*/
#pragma omp parallel for private(i, j, jj)
  for (i = 0; i < coo_matrix->nrows; i++) {
    int col = 0;
    // int elem = freq[i];
    // int dum = 0;

    for (jj = freq[i]; jj < max_jmp; jj++) {
      for (j = 0; j < freq[i]; j++) {
        if (s_coo_matrix->columns[i * max_jmp + j] == col) {
          col++;
          j = 0;
        }
      }
      s_coo_matrix->columns[i * max_jmp + jj] = col;
      col++;
    }

    // for (j = 0; j < max_jmp - freq[i] && col < s_coo_matrix->ncolumns; j++)
    // {
    //     if (s_coo_matrix->columns[i * max_jmp + j + dum] != col)
    //     {
    //         s_coo_matrix->columns[i * max_jmp + elem] = col;
    //         col++;
    //         elem++;
    //     }
    //     else
    //     {
    //         col++;
    //         dum++;
    //         j--;
    //     }
    // }
  }

  printf("End of sorting with padded row %d\n", max_jmp);
}

void convert_coo_to_csr(const SparseMatrixCOO *coo_matrix,
                        SparseMatrixCSR *csr_matrix, int free_coo) {

  /* Allocate CSR Matrix data structure in memory */
  csr_matrix->row_pointers =
      (int *)aligned_alloc(align_size, (coo_matrix->nrows + 1) * sizeof(int));
  memset(csr_matrix->row_pointers, 0, (coo_matrix->nrows + 1) * sizeof(int));

  csr_matrix->column_indices =
      (int *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(int));

  csr_matrix->values =
      (double *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(double));

  // Store the number of Non-Zero elements in each Row
  for (unsigned int i = 0; i < coo_matrix->nnz; i++)
    csr_matrix->row_pointers[coo_matrix->rows[i]]++;

  // Update Row Pointers so they consider the previous pointer offset
  // (using accumulative sum).
  unsigned int cum_sum = 0;
  for (unsigned int i = 0; i < coo_matrix->nrows; i++) {
    unsigned int row_nnz = csr_matrix->row_pointers[i];
    csr_matrix->row_pointers[i] = cum_sum;
    cum_sum += row_nnz;
  }

  /*  Adds COO values to CSR

      Note: Next block of code reuses csr->row_pointers[] to keep track of the
     values added from the COO matrix. This way is able to create the CSR matrix
     even if the COO matrix is not ordered by row. In the process, it 'trashes'
     the row pointers by shifting them one position up. At the end, each
     csr->row_pointers[i+1] should be in csr->row_pointers[i] */

  for (unsigned int i = 0; i < coo_matrix->nnz; i++) {
    unsigned int row_index = coo_matrix->rows[i];
    unsigned int column_index = coo_matrix->columns[i];
    double value = coo_matrix->values[i];

    unsigned int j = csr_matrix->row_pointers[row_index];
    csr_matrix->column_indices[j] = column_index;
    csr_matrix->values[j] = value;
    csr_matrix->row_pointers[row_index]++;
  }

  // Restore the correct row_pointers
  for (unsigned int i = coo_matrix->nrows - 1; i > 0; i--) {
    csr_matrix->row_pointers[i] = csr_matrix->row_pointers[i - 1];
  }
  csr_matrix->row_pointers[0] = 0;
  csr_matrix->row_pointers[coo_matrix->nrows] = coo_matrix->nnz;

  csr_matrix->nnz = coo_matrix->nnz;
  csr_matrix->nrows = coo_matrix->nrows;
  csr_matrix->ncolumns = coo_matrix->ncolumns;

  /*  For each row, sort the corresponding arrasy csr.column_indices and
     csr.values

      TODO: We should check if this step makes sense or can be optimized
      1) If the .mtx format by definition is ordered
      2) If we force the COO Matrix to be ordered first, we can avoid this
      3) Test speed of standard library sorting vs current sorting approach. */

  // for (unsigned int i = 0; i < csr_matrix->nrows; i++)
  // {
  //     // print_arr_uint(csr_matrix->row_pointers[i+1]-
  //     csr_matrix->row_pointers[i], "before",
  //     &csr_matrix->column_indices[csr_matrix->row_pointers[i]]);

  //     // This could be optimized
  // sort_paired_vectors(csr_matrix->row_pointers[i], csr_matrix->row_pointers[i
  // + 1],
  //                     csr_matrix->column_indices, csr_matrix->values);

  //     // fprintf(stderr, "Sorting from: [%" PRIu64 "] to [%" PRIu64 "]\n",
  //     csr_matrix->row_pointers[i],  csr_matrix->row_pointers[i + 1]);

  //     // radix_sort_paired_vectors(csr_matrix->column_indices,
  //     csr_matrix->values, csr_matrix->row_pointers[i],
  //     csr_matrix->row_pointers[i + 1]);
  //     // print_arr_uint(csr_matrix->row_pointers[i+1]-
  //     csr_matrix->row_pointers[i], "after",
  //     &csr_matrix->column_indices[csr_matrix->row_pointers[i]]);
  //     // exit(0);
  // }

  if (free_coo) {
    free(coo_matrix->values);
    free(coo_matrix->rows);
    free(coo_matrix->columns);
  }
}
size_t get_multiple_of_align_size(size_t size) {
  // Returns the closest multiple of the <align_size> to <size>
  size_t padded_size = ((align_size - (size % align_size)) % align_size) + size;
  return padded_size;
}
