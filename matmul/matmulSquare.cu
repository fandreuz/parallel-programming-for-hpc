#define N 10
#define THREADS_PER_BLOCK 20

#ifdef DEBUG
#include "stdio.h"
#endif

__global__ void matmul(int *a, int *b, int *c) {
  int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (c_idx >= N * N) {
    return;
  }

  int *a_row = a + (c_idx / N) * N;
  int *b_col = b + c_idx % N;
  int *c_write = c + c_idx;

  for (int k = 0; k < N; ++k)
    *c_write += a_row[k] * b_col[k * N];
}

int main(void) {
  int array_size = N * N;
  int array_memory_size = array_size * sizeof(int);

  int *a = (int *)malloc(array_memory_size);
  for (int i = 0; i < array_size; ++i) {
    if (i % N == 0)
      a[i] = i / N;
    else
      a[i] = 0;
  }
  int *b = (int *)malloc(array_memory_size);
  for (int i = 0; i < array_size; ++i) {
    if (i / N == i % N)
      b[i] = 1;
    else
      b[i] = 0;
  }

#ifdef DEBUG
  printf("a:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d ", a[i * N + j]);
    }
    printf("\n");
  }
  printf("b:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d ", b[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  int *dev_a;
  cudaMalloc((void **)&dev_a, array_memory_size);
  cudaMemcpy(dev_a, a, array_memory_size, cudaMemcpyHostToDevice);

  int *dev_b;
  cudaMalloc((void **)&dev_b, array_memory_size);
  cudaMemcpy(dev_b, b, array_memory_size, cudaMemcpyHostToDevice);

  int *dev_c;
  cudaMalloc((void **)&dev_c, array_memory_size);

  matmul<<<(array_size / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(
      dev_a, dev_b, dev_c);

  int *c = (int *)malloc(array_memory_size);
  cudaMemcpy(c, dev_c, array_memory_size, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("c:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d ", c[i * N + j]);
    }
    printf("\n");
  }
#endif

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  free(a);
  free(b);
  free(c);

  return 0;
}