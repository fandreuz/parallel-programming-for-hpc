#include "stdio.h"

// each block takes care of a row of the transposed matrix
__global__ void tranpose_kernel(int *a, int *aT) {
  int aT_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int a_idx = threadIdx.x * gridDim.x + blockIdx.x;
  aT[aT_idx] = a[a_idx];
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

  int *dev_aT;
  cudaMalloc((void **)&dev_aT, array_memory_size);

  tranpose_kernel<<<N, THREADS_PER_BLOCK>>>(dev_a, dev_aT);

  cudaMemcpy(a, dev_aT, array_memory_size, cudaMemcpyDeviceToHost);

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
  cudaFree(dev_aT);

  free(a);

  return 0;
}