# Parallel programming for HPC
Weekly exercises for the course in _Parallel programming for HPC_ @ UniTS.

## Topics
- MPI
- OpenMP
- Theory of GPU programming
- CUDA
- OpenACC

## Some plots
The following time measurements were taken on standard nodes on [Marconi-100](https://wiki.u-gov.it/confluence/pages/viewpage.action?pageId=336727645).

### MatMul (MPI, CUDA, cuBLAS)
Distributed matrix multiplication ([matmul](matmul)) with MPI, BLAS and cuBLAS.

<ins>**2500x2500**</ins>

![](imgs/matmul_2500.png)

<ins>**5000x5000**</ins>

![](imgs/matmul_5000.png)

### Jacobi (MPI, OpenACC)

![](imgs/jacobi_10000_1000.png)

### FFT 3D (FFTW, `MPI_Alltoallv`)

![](imgs/fft3d.png)