/*
 *   Floyd's all-pairs shortest path algorithm
 *   (Checkerboard decomposition version)
 *
 *   Given an nxn matrix of distances between pairs of
 *   vertices, this MPI program computes the shortest path
 *   between every pair of vertices in parallel.
 *
 *   CMSC5702 Assignment 2
 *   =====================
 *   Student Name: <Your full name here>
 *   Student ID: <Your student ID here>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <mpi.h>
#include "MyMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT

#define INF 999999

/* Safe add to avoid overflow when combining two finite path lengths */
static inline dtype safe_add(dtype x, dtype y) {
    long long s = (long long)x + (long long)y;
    if (s > (long long)INT_MAX) return INT_MAX;
    if (s < (long long)INT_MIN) return INT_MIN;
    return (dtype)s;
}

void compute_shortest_paths(dtype**, int, int, MPI_Comm);

int main(int argc, char *argv[]) {
    dtype** a;           /* Doubly-subscripted array */
    dtype*  storage;     /* Local portion of array elements */
    int     id;          /* Process rank */
    int     m;           /* Rows in matrix */
    int     n;           /* Columns in matrix */
    int     p;           /* Number of processes */
    double  time, max_time;
    MPI_Comm cart_comm;  /* Cartesian topology communicator */
    // TODO: add more variables as you see fit.
    int dims[2], periods[2], reorder;
    int q;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (id == 0 && argc != 2) {
        printf("Usage: %s <input matrix file>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // TODO: add code to create the Cartesian topology communicator.
    // Assume it is named cart_comm for the following code to work.
    q = (int)(sqrt((double)p));
    if (q * q != p) {
        if (id == 0) {
            printf("Error: Number of processes must be a perfect square\n");
            fflush(stdout);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    dims[0] = q;
    dims[1] = q;
    periods[0] = 0;
    periods[1] = 0;
    reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    if (cart_comm == MPI_COMM_NULL) {
        if (id == 0) {
            fprintf(stderr, "Error: MPI_Cart_create returned MPI_COMM_NULL\n");
            fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    read_checkerboard_matrix(argv[1], (void *) &a,
        (void *) &storage, MPI_TYPE, &m, &n, cart_comm);

    if (m != n) terminate(id, "Matrix must be square\n");

#ifdef DEBUG
    /* Verify the initial weight matrix */
    print_checkerboard_matrix((void **)a, MPI_TYPE, m, n, cart_comm);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    time = -MPI_Wtime();
    /* Run the Floyd's algorithm */
    compute_shortest_paths((dtype **)a, m, n, cart_comm);
    time += MPI_Wtime();
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD);

    if (!id)
        printf("Checkerboard Floyd, matrix size %d, %d processes: %.5f seconds\n",
            n, p, max_time);

#ifdef DEBUG
    /* Verify the resultant weight matrix */
    print_checkerboard_matrix((void **) a, MPI_TYPE, m, n, cart_comm);
#endif

    MPI_Finalize();
    return 0;
}

/*
 * This function implements the Floyd's algorithm to compute the shortest
 * path between every pair of vertices in adjacency matrix a for an input
 * size m*n, which is decomposed in a checkerboard manner. You can pass 
 * a Cartesian topology communicator via the grid_comm parameter and 
 * determine in this function the block of the matrix that should be 
 * computed by this process. 
 */
void compute_shortest_paths(dtype **a, int m, int n, MPI_Comm grid_comm) {

    int grid_size[2], grid_coords[2], grid_periods[2];
    int q;
    int local_rows, local_cols;
    int i, j, k;
    int id;

    MPI_Comm row_comm, col_comm;
    int remain_dims[2];

    MPI_Group grid_group, row_group, col_group;

    dtype *k_row;
    dtype *k_col;

    /* Retrieve topology info for this process in the 2D grid */
    MPI_Comm_rank(grid_comm, &id);
    MPI_Cart_get(grid_comm, 2, grid_size, grid_periods, grid_coords);
    q = grid_size[0];

    /* Determine local block size on this process */
    local_rows = BLOCK_SIZE(grid_coords[0], q, m);
    local_cols = BLOCK_SIZE(grid_coords[1], q, n);

    /* Create row-based and column-based subcommunicators */
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(grid_comm, remain_dims, &row_comm);

    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid_comm, remain_dims, &col_comm);

    MPI_Comm_group(grid_comm, &grid_group);
    MPI_Comm_group(row_comm,  &row_group);
    MPI_Comm_group(col_comm,  &col_group);

    k_row = (dtype *)malloc((size_t)local_cols * sizeof(dtype));
    k_col = (dtype *)malloc((size_t)local_rows * sizeof(dtype));

    if (k_row == NULL || k_col == NULL) {
        fprintf(stderr, "Rank %d: malloc failed\n", id);
        fflush(stderr);
        MPI_Abort(grid_comm, -1);
    }

    for (k = 0; k < n; k++) {

        int owner_row = BLOCK_OWNER(k, q, m);
        int owner_col = BLOCK_OWNER(k, q, n);

        int local_k_row = k - BLOCK_LOW(owner_row, q, m);
        int local_k_col = k - BLOCK_LOW(owner_col, q, n);

        /* Broadcast a row to all processes in the same column */
        {
            int root_coords[2];
            int root_grid_rank, root_col_rank;

            root_coords[0] = owner_row;
            root_coords[1] = grid_coords[1];
            MPI_Cart_rank(grid_comm, root_coords, &root_grid_rank);
            MPI_Group_translate_ranks(grid_group, 1, &root_grid_rank,
                                      col_group, &root_col_rank);

            if (grid_coords[0] == owner_row) {
                for (j = 0; j < local_cols; j++)
                    k_row[j] = a[local_k_row][j];
            }
            MPI_Bcast(k_row, local_cols, MPI_TYPE, root_col_rank, col_comm);
        }

        /* Broadcast a column to all processes in the same row */
        {
            int root_coords[2];
            int root_grid_rank, root_row_rank;

            root_coords[0] = grid_coords[0];
            root_coords[1] = owner_col;
            MPI_Cart_rank(grid_comm, root_coords, &root_grid_rank);
            MPI_Group_translate_ranks(grid_group, 1, &root_grid_rank,
                                      row_group, &root_row_rank);

            if (grid_coords[1] == owner_col) {
                for (i = 0; i < local_rows; i++)
                    k_col[i] = a[i][local_k_col];
            }
            MPI_Bcast(k_col, local_rows, MPI_TYPE, root_row_rank, row_comm);
        }

        /* Compute all-pairs shortest paths locally */
        for (i = 0; i < local_rows; i++) {
            for (j = 0; j < local_cols; j++) {
                if (k_col[i] != INF && k_row[j] != INF) {
                    dtype candidate = safe_add(k_col[i], k_row[j]);
                    a[i][j] = MIN(a[i][j], candidate);
                }
            }
        }
    }

    free(k_row);
    free(k_col);
    MPI_Group_free(&grid_group);
    MPI_Group_free(&row_group);
    MPI_Group_free(&col_group);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}
