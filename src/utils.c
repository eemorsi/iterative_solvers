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

#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <string.h>
#include "macros.h"
#include <assert.h>

int validate_vector(const double *y, const double *y_ref, const size_t size)
{
    /*
        Input two vectors that are supposed to have the same content.
        Useful to compare floating point accuracy deviation.
        If the different 
    */
    int debug = 1;
    int is_100_correct = 1, is_good_enough = 1, correct = 0;
    double abs_diff, rel_diff;
    double threshold = 1e-7;
    int max_errors = 50;
    int err_count = 0;

    int v = 0;
    while (v < size)
    {
        correct = y_ref[v] == y[v];

        if (!correct || (y[v] != y[v])) // if y is NaN y != y will be true
        {
            is_100_correct = 0;

            abs_diff = y_ref[v] - y[v];
            rel_diff = abs_diff / y_ref[v];
            if (fabs(rel_diff) > threshold || (y[v] != y[v]))
            {
                is_good_enough = 0;
                fprintf(stderr, "Warning: element in y[%d] has a relative diff = <%f>!\n", v, rel_diff);
                if (debug)
                    fprintf(stderr, "   yref: <%g> != <%g>\n", y_ref[v], y[v]);

                err_count++;
                if (err_count > max_errors)
                    return 0;
            }
        }

        v++;
    }

    // Return 2 if 100% correct, 1 if good enough, 0 if not correct
    return is_100_correct + is_good_enough;
}

void init_x(double *x, size_t n, int test_case)
{
    size_t i;
    switch (test_case)
    {
    case 1:
        for (i = 0; i < n; i++)
            x[i] = 3.0;
        break;
    case 2:
        for (i = 0; i < n; i++)
            x[i] = (double)i + 1;
        // x[i] = (double) 1.0;
        break;
    default:
        printf("Unexpected X Initialization\n");
        exit(-1);
    }
}

void memset_float(double *vec, float value, size_t n)
{
    for (size_t i = 0; i < n; i++)
        vec[i] = value;
}

void print_arr_uint(size_t N, char *name, size_t *vector)
{
    size_t i;
    fprintf(stderr, "\nPrinting vector: %s\n", name);
    for (i = 0; i < N; i++)
    {
        if (!i % 30)
            fprintf(stderr, "\n");
        fprintf(stderr, "%lu, ", vector[i]);
    }
    fprintf(stderr, "\n");
}

void print_arr_float(size_t N, char *name, double *vector)
{
    size_t i;
    printf("\nPrinting vector: %s", name);
    for (i = 0; i < N; i++)
    {
        if (!i % 30)
            printf("\n");
        printf("%g, ", vector[i]);
    }
    printf("\n");
}

void check_mem_alloc(void *ptr, const char *err_msg)
{
    if (ptr == NULL)
    {
        fprintf(stderr, "Memory Allocation Error: could not allocate %s. Application will exit.", err_msg);
        exit(1);
    }
}

size_t *get_rows_size(const size_t *__restrict row_ptrs, const size_t nrows, size_t * padded_size)
{
    size_t *rs = (size_t *)aligned_alloc(align_size, get_multiple_of_align_size(nrows * sizeof(size_t)));
    check_mem_alloc(rs, "rows_size\n");

    // size_t vsize = get_multiple_of_align_size(((nrows + 255) / 256) * 256 * sizeof(size_t));
    // assert((vsize % 2048) == 0);
    for (size_t i = 0; i < nrows; i++)
    {
        rs[i] = row_ptrs[i + 1] - row_ptrs[i];
    }

    return rs;
}

size_t **get_rows_size_perblock(const SparseMatrixCSR *csr_matrix, const uint64_t num_blocks)
{
    size_t block_size = (csr_matrix->ncolumns + num_blocks - 1) / num_blocks;
#ifdef SPMV_DEBUG
    fprintf(stderr, "The X vector block width is: %lu elements\n", block_size);
#endif

    size_t vsize = get_multiple_of_align_size(((csr_matrix->nrows + 255) / 256) * 256 * sizeof(size_t));
    assert((vsize % 2048) == 0);

    size_t *rs = (size_t *)aligned_alloc(align_size, get_multiple_of_align_size(num_blocks * vsize));
    memset(rs, 0, num_blocks * vsize); // To take into account padded 0 final slice rows.

    size_t **row_sizes = (size_t **)aligned_alloc(align_size, get_multiple_of_align_size(num_blocks * sizeof(size_t *)));
    check_mem_alloc(row_sizes, "row_sizes\n");

    // Init Array of Arrays pointers
    for (size_t b = 0; b < num_blocks; b++)
    {
        row_sizes[b] = &rs[b * csr_matrix->nrows];
        memset(row_sizes[b], 0, vsize);
    }
    // This could be improved with some kind of binary search?
    for (size_t i = 0; i < csr_matrix->nrows; i++)
    {
        /* For each row, find the block where the element belongs in the new blocked matrix and add it to the size count. */
        for (size_t j = csr_matrix->row_pointers[i]; j < csr_matrix->row_pointers[i + 1]; j++)
        {
            size_t col_idx = csr_matrix->column_indices[j];
            size_t block_idx = col_idx / block_size;
            row_sizes[block_idx][i]++;
        }
        // fprintf(stderr, "Row [%" PRIu64 "] has size [%" PRIu64 "]\n", i, sizes[i]);
    }

    return row_sizes;
}

size_t get_multiple_of_align_size(size_t size)
{
    // Returns the closest multiple of the <align_size> to <size>
    size_t padded_size = ((align_size - (size % align_size)) % align_size) + size;

#ifdef SPMV_DEBUG
    fprintf(stderr, "Allocating [%" PRIu64 "] bytes, should be aligned to: [%" PRIu64 "]\n", padded_size, align_size);
#endif

    return padded_size;
}

size_t get_num_verticalops(const size_t *__restrict vrows_size, const size_t nrows, const int vlen, size_t *__restrict slice_width)
{
    // Pre: rows_size is ordered in descent in blocks of vlen size. Meaning every multiple of 256 in rows_size,
    //      including 0, contains the max row size of that block of rows.
    //      e.g: rows_size[0] contains the highest row size between rows_size[0] to rows_size[0+vlen-1];

    // Post:  Returns the total number of vertical ops required. We need this to allocate the vactive_lanes.

    size_t total_vops = 0;
    size_t vb_idx = 0;
    for (size_t i = 0; i < nrows; i += vlen)
    {
        total_vops += vrows_size[i];
        slice_width[vb_idx++] = vrows_size[i];
    }

    return total_vops;
}

void set_active_lanes(const size_t *__restrict vrows_size, const size_t nrows,
                      const uint32_t vlen, uint8_t *__restrict vactive_lanes)
{
    size_t vactive_idx = 0;

    for (int64_t i = 0; i < nrows; i += vlen)
    {
        // last_row = min(nrows, i+vlen) - 1;
        size_t last_row = ((i + vlen) > nrows) ? (nrows - (uint64_t)1) : (i + vlen - 1);

        // prev_rsize = 0 causes to write first as many vactive_lanes as the minimum row size (which is at vrow_size[end]).
        size_t prev_rsize = 0;
        size_t lanes_to_disable = 0;

        // current_lanes = min(vlen, end - i); NOTE: You must use current lanes + 1 on the solver.
        uint8_t current_lanes = (uint8_t)(last_row - i);

        // For each row in this slice:
        for (int64_t j = last_row; j >= i; j--)
        {
            // Traverse backwards
            size_t current_rsize = vrows_size[j];
            size_t diff = current_rsize - prev_rsize;

            if (diff > 0)
            {
                current_lanes -= lanes_to_disable;
                DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 "): Adding [%" PRIu64 "-%" PRIu64 "] vops using [%" PRIu8 "] lanes\n", j, current_rsize, vactive_idx, vactive_idx + diff, current_lanes + 1));
                for (size_t k = 0; k < diff; k++)
                {
                    vactive_lanes[vactive_idx++] = current_lanes;
                }

                lanes_to_disable = 1;
                prev_rsize = current_rsize;
            }
            else
            {
                DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 ") wont add anything.\n", j, current_rsize));
                lanes_to_disable++;
            }
        }
    }
}

void set_slice_vop_length(const size_t *__restrict rows_size, const size_t slice_height, uint8_t *__restrict vop_lengths)
{
    size_t vactive_idx = 0;

    size_t last_row = slice_height - 1;
    size_t prev_rsize = 0;
    size_t lanes_to_disable = 0;

    uint8_t current_lanes = slice_height - 1; // NOTE: You must use current lanes + 1 on the solver.

    for (int64_t j = last_row; j >= 0; j--)
    {
        // Traverse backwards
        size_t rsize = rows_size[j];
        size_t diff = rsize - prev_rsize;

        if (diff > 0)
        {
            current_lanes -= lanes_to_disable;
            DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 "): Adding [%" PRIu64 "-%" PRIu64 "] vops using [%" PRIu8 "] lanes\n", j, current_rsize, vactive_idx, vactive_idx + diff, current_lanes + 1));
            // Insert VOPS with the same vop lengths
            for (size_t k = 0; k < diff; k++)
            {
                vop_lengths[vactive_idx++] = current_lanes;
            }

            lanes_to_disable = 1;
            prev_rsize = rsize;
        }
        else
        {
            DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 ") wont add anything.\n", j, current_rsize));
            lanes_to_disable++;
        }
    }
}
