/* Author: Constantino Gómez, 2020
 *
 * Licensed to the Barcelona Supercomputing Center (BSC) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The BSC licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UTILS_H
#define UTILS_H

#include "sparse_matrix.h"

int validate_vector(const double *y, const double *y_ref, const size_t size);
void init_x(double *x, size_t n, int test_case);
void memset_float(double *vec, float value, size_t n);
void print_arr_float(size_t N, char *name, double *vector);
void print_arr_uint(size_t N, char *name, size_t *vector);
void check_mem_alloc(void * ptr, const char * err_msg);


size_t *get_rows_size(const size_t *__restrict row_ptrs, const size_t nrows, size_t * padded_size);
size_t **get_rows_size_perblock(const SparseMatrixCSR *csr_matrix, const uint64_t num_blocks);

size_t get_multiple_of_align_size(size_t size);
size_t get_num_verticalops(const size_t *__restrict vrows_size, const size_t nrows, const int vlen, size_t *__restrict vblock_size);
size_t get_num_verticalops_blocked(size_t **__restrict vrows_size, const size_t nrows, const int vlen, size_t *__restrict vblock_size, const size_t num_blocks);

void set_active_lanes(const size_t *__restrict vrows_size, const size_t nrows, const uint32_t vlen, uint8_t *__restrict vactive_lanes);
void set_slice_vop_length(const size_t *__restrict rows_size, const size_t slice_height, uint8_t *__restrict vop_lengths);

#endif // UTILS_H
