/*
 *   Floyd's all-pairs shortest path algorithm
 *   (Sequential version)
 *
 *   Given an nxn matrix of distances between pairs of
 *   vertices, this program computes the shortest path
 *   between every pair of vertices.
 *
 *   Written by CMSC5702 teaching team.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef int dtype;

void alloc_matrix(void ***, int, int, int);
void print_matrix(int**, int);
void compute_shortest_paths(int**, int);

int main(int argc, char *argv[]) {
    int     m;                    /* Rows in matrix */
    int     n;                    /* Columns in matrix */
    dtype** a;                    /* Two-dimensional array */
    struct timespec stime, etime; /* Start and end times */
    FILE    *in;                  /* Input data file */

    /* Load the adjacency matrix from a data file */
    if ((in = fopen(argv[1], "r")) == NULL) {
        printf("Open file error\n");
        exit(-1);
    }
    fread(&m, sizeof(int), 1, in);
    fread(&n, sizeof(int), 1, in);
    alloc_matrix((void ***)&a, m, n, sizeof(dtype));
    for (int i = 0; i < m; i++) {
        fread(a[i], sizeof(dtype), n, in);
    }
    fclose(in);

    if (m != n) {
        printf("Matrix must be square\n");
        exit(-1);
    }

#ifdef DEBUG
    /* Verfiy the initial weight matrix */
    print_matrix((dtype **)a, n);
#endif

    timespec_get(&stime, TIME_UTC);

    /* Run the Floyd's algorithm */
    compute_shortest_paths((dtype **)a, n);

    timespec_get(&etime, TIME_UTC);

    printf("Sequential Floyd, matrix size %d: %.5Lf seconds\n", n,
        (long double)(etime.tv_sec - stime.tv_sec) +
        (long double)(etime.tv_nsec - stime.tv_nsec) / 1000000000.0L
    );

#ifdef DEBUG
    print_matrix(a, n);
#endif
}

/* The Floyd's algorithm */
void compute_shortest_paths(dtype **a, int n) {
   for (int k = 0; k < n; k++)
      for (int i = 0; i < n; i++)
         for (int j = 0; j < n; j++)
            a[i][j] = MIN(a[i][j], a[i][k] + a[k][j]);
}

/* Print the weight matrix on screen */
void print_matrix (dtype **a, int n) {
    char str[8];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sprintf(str, "%d", a[i][j]);
            printf("%-8s", str);
        }
        printf("\n");
    }
}

/* Allocate a two-dimensional array with 'm' rows and 'n' colums,
 * where each entry occupies 'size' bytes */
void alloc_matrix(void ***a, int m, int n, int size) {
    void *storage;
    storage = (void *)malloc(m * n * size);
    *a = (void **)malloc(m * sizeof(void *));
    for (int i = 0; i < m; i++) {
        (*a)[i] = storage + i * n * size;
    }
}