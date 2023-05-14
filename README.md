# Parallel programming for HPC
Weekly exercises for the course in _Parallel programming for HPC_ @ UniTS.

## Topics
- Distributed parallelism (MPI)
- BLAS
- (NVIDIA) GPU programming
    - CUDA
    - Theory & best practices
- cuBLAS
- OpenACC
- FFTW

## Some plots
The following time measurements were taken on standard nodes on [Marconi-100](https://wiki.u-gov.it/confluence/pages/viewpage.action?pageId=336727645).

### Matrix multiplication (MPI, BLAS, cuBLAS)

<ins>**2500x2500**</ins>

![](imgs/matmul_2500.png)

<ins>**5000x5000**</ins>

![](imgs/matmul_5000.png)

### Jacobi method --- heat diffusion (MPI, OpenACC)

![](imgs/jacobi_10000_1000.png)

### FFT 3D (MPI, FFTW)

![](imgs/fft3d.png)