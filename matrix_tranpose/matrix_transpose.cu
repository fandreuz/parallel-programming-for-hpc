#include "stdio.h"

// each block takes care of a row of the transposed matrix
__global__ void tranpose_kernel(int *a) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx / N;
  int j = idx % N;
  if (i < j) {
    int temp = a[i * N + j];
    a[i * N + j] = a[j * N + i];
    a[j * N + i] = temp;
  }
}

int main(void) {
  int array_size = N * N;
  int array_memory_size = array_size * sizeof(int);

  int *a = (int *)malloc(array_memory_size);
  for (int i = 0; i < array_size; ++i) {
    a[i] = 0;
  }
  a[array_size - N] = 1;

#ifdef DEBUG
  printf("Input:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d ", a[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  int *dev_a;
  cudaMalloc((void **)&dev_a, array_memory_size);
  cudaMemcpy(dev_a, a, array_memory_size, cudaMemcpyHostToDevice);
  tranpose_kernel<<<N * N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(dev_a);

  cudaMemcpy(a, dev_a, array_memory_size, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("Output:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d ", a[i * N + j]);
    }
    printf("\n");
  }
#endif

  cudaFree(dev_a);
  free(a);

  return 0;
}