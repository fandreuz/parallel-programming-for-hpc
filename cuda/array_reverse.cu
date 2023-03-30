#define N 100
#define THREADS_PER_BLOCK 10
#include "stdio.h"

__global__ void array_reverse(int *a, int *aR) {
  int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int temp = a[a_idx];
  aR[N - 1 - a_idx] = temp;
}

int main(void) {
  int array_size = N;
  int array_memory_size = array_size * sizeof(int);

  int *a = (int *)malloc(array_memory_size);
  for (int i = 0; i < array_size; ++i)
    a[i] = i;

#ifdef DEBUG
  printf("Input:\n");
  for (int i = 0; i < N; ++i)
    printf("%d ", a[i]);
  printf("\n");
#endif

  int *dev_a;
  cudaMalloc((void **)&dev_a, array_memory_size);
  cudaMemcpy(dev_a, a, array_memory_size, cudaMemcpyHostToDevice);

  int *dev_aT;
  cudaMalloc((void **)&dev_aT, array_memory_size);

  array_reverse<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_a, dev_aT);

  cudaMemcpy(a, dev_aT, array_memory_size, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("Output:\n");
  for (int i = 0; i < N; ++i)
    printf("%d ", a[i]);
  printf("\n");
#endif

  cudaFree(dev_a);
  cudaFree(dev_aT);

  free(a);

  return 0;
}