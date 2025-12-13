

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
#include <mpi.h>
#include "MyMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT

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

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (id == 0 && argc != 2) {
        printf("Usage: %s <input matrix file>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);  // Abort all processes with error code -1
    }

    // TODO: add code to create the Cartesian topology communicator.
    // Assume it is named cart_comm for the following code to work.

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
    /*
     *  TODO: Add your code to complete the algorithm.
     *
     *  Example tasks:
     *  - Retrieve the grid topology information.
     *  - Determine # rows and columns held by this process.
     *  - You may split the grid further into row-based and column-based
     *    communicators for sub-group broadcast of data if you see fit.
     *  - Broadcast a row to all processes in the same column.
     *  - Broadcast a column to all processes in the same row.
     *  - Compute all-pairs shortest paths locally.
     *
     * Reminder: Free the all buffers you might dynamically allocated
     *           before leaving. Also better remove all the provided
     *           TODO comments, write your own comments instead.
     * 
     */
}