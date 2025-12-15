/*
 *   Floyd's all-pairs shortest path algorithm
 *   (CUDA version)
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
#include <cuda_runtime.h>
#include <time.h>

/* Simple CUDA error checking helper */
#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));   \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while (0)

#define BLOCK_SIZE 32

typedef int dtype;

/* This function is the CUDA kernel code to be run on the GPGPU. */
__global__ void compute_shortest_paths(dtype *d_a, int n, int k) {
    // TODO: Add your code to complete the Floyd's algorithm.
    //       You may use the built-in CUDA min() function here.

    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (i >= n || j >= n) return;

    const dtype INF = 999999;

    dtype a_ik = d_a[i * n + k];
    dtype a_kj = d_a[k * n + j];

    // If either segment is INF, the path via k is not usable.
    if (a_ik == INF || a_kj == INF) return;

    dtype via = a_ik + a_kj;

    dtype cur = d_a[i * n + j];
    d_a[i * n + j] = min(cur, via);
}

void print_matrix(dtype *a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6d ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    FILE *in;                      /* Input data file */
    int m, n;                      /* Rows and columns in matrix */
    dtype *a, *d_a;                /* Host and device arrays */
    struct timespec stime, etime;  /* Start and end times */

    /* Ensure the input file path is provided. */
    if (argc != 2) {
        printf("Usage: %s <input matrix file>\n", argv[0]);
        return -1;
    }

    /* Load the adjacency matrix from a binary file. */
    if ((in = fopen(argv[1], "rb")) == NULL) {
        printf("File open error!\n");
        return -1;
    }
    (void)!fread(&m, sizeof(int), 1, in);
    (void)!fread(&n, sizeof(int), 1, in);
    a = (dtype *)malloc(m * n * sizeof(dtype));
    (void)!fread(a, sizeof(dtype), m * n, in);
    fclose(in);

    if (m != n) {
        printf("Matrix must be square\n");
        free(a);
        return -1;
    }

#ifdef DEBUG
    printf("Initial weight matrix:\n");
    print_matrix(a, n);
#endif

    // Allocate the adjacency matrix (flattened) on the GPU device
    // Assume the array is called d_a.
    CUDA_CHECK(cudaMalloc(&d_a, m * n * sizeof(dtype)));

    // TODO: copy the adjacency matrix from host to device
    CUDA_CHECK(cudaMemcpy(d_a, a, m * n * sizeof(dtype), cudaMemcpyHostToDevice));

    timespec_get(&stime, TIME_UTC);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

    for (int k = 0; k < n; k++) {
        // Pass n and k to the kernel
        compute_shortest_paths<<<blocks, threads>>>(d_a, n, k);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Check for potential errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }

    timespec_get(&etime, TIME_UTC);

    // TODO: copy the adjacency matrix from device to host
    CUDA_CHECK(cudaMemcpy(a, d_a, m * n * sizeof(dtype), cudaMemcpyDeviceToHost));

    printf("CUDA Floyd, matrix size %d: %.5Lf seconds\n", n,
        (long double)(etime.tv_sec - stime.tv_sec) +
        (long double)(etime.tv_nsec - stime.tv_nsec) / 1000000000.0L);

#ifdef DEBUG
    printf("Shortest-path matrix:\n");
    print_matrix(a, n);
#endif

    CUDA_CHECK(cudaFree(d_a));
    free(a);
    return 0;
}