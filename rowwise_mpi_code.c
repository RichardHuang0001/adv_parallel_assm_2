/*
 *   Floyd's all-pairs shortest path algorithm
 *   (Rowwise decomposition version)
 *
 *   Given an nxn matrix of distances between pairs of
 *   vertices, this MPI program computes the shortest path
 *   between every pair of vertices in parallel.
 *
 *   Originally written by Michael J. Quinn 
 *                         (in his MPI and OpenMP book, ch.6).
 *   Modified by CMSC5702 teaching team.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "MyMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT

int main(int argc, char *argv[]) {
    dtype** a;         /* Doubly-subscripted array */
    dtype*  storage;   /* Local portion of array elements */
    int     id;        /* Process rank */
    int     m;         /* Rows in matrix */
    int     n;         /* Columns in matrix */
    int     p;         /* Number of processes */
    double  time, max_time;

    void compute_shortest_paths(int, int, int**, int);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    read_row_striped_matrix(argv[1], (void *) &a,
        (void *) &storage, MPI_TYPE, &m, &n, MPI_COMM_WORLD);

    if (m != n) terminate(id, "Matrix must be square\n");

#ifdef DEBUG
    /* Verify the initial weight matrix */
    print_row_striped_matrix((void **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    time = -MPI_Wtime();
    /* Run the Floyd's algorithm */
    compute_shortest_paths(id, p, (dtype **) a, n);
    time += MPI_Wtime();
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD);

    if (!id)
        printf("Rowwise Floyd, matrix size %d, %d processes: %.5f seconds\n",
            n, p, max_time);

#ifdef DEBUG
    /* Verify the resultant weight matrix */
    print_row_striped_matrix((void **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);
#endif

    MPI_Finalize();
}

/*
 * This function implements the Floyd's algorithm to compute the shortest
 * path between every pair of vertices in adjacency matrix a for an input
 * size n, which is decomposed rowwise by p processes, where id is the
 * rank of the process calling this function.
 */
void compute_shortest_paths(int id, int p, dtype **a, int n)
{
    int  i, j, k;
    int  offset;   /* Local index of broadcast row */
    int  root;     /* Process controlling row to be bcast */
    int* tmp;      /* Holds the broadcast row */

    tmp = (dtype *) malloc(n * sizeof(dtype));
    for (k = 0; k < n; k++) {
        root = BLOCK_OWNER(k, p, n);
        if (root == id) {
            offset = k - BLOCK_LOW(id, p, n);
            for (j = 0; j < n; j++)
                tmp[j] = a[offset][j];
        }
        MPI_Bcast(tmp, n, MPI_TYPE, root, MPI_COMM_WORLD);
        for (i = 0; i < BLOCK_SIZE(id, p, n); i++)
            for (j = 0; j < n; j++)
                a[i][j] = MIN(a[i][j], a[i][k] + tmp[j]);
    }
    free (tmp);
}