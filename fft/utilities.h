#pragma once

#include <complex.h>
#include <stdlib.h>
#include <fftw3.h>
#include <mpi.h>
#include <string.h>

struct Fft3dInfo {
  int n1;
  int loc_n1;
  int loc_n1_offset;

  int n2;

  int n3;
  int loc_n3;

  fftw_plan fft_2d_many;
  fftw_plan fft_1d_many;
  fftw_plan ifft_2d_many;
  fftw_plan ifft_1d_many;

  fftw_complex *fft_2d_in;
  fftw_complex *fft_2d_out;
  fftw_complex *fft_1d_in;
  fftw_complex *fft_1d_out;

  int *send_counts;
  int *send_displacements;
  int *recv_counts;
  int *recv_displacements;
};

typedef struct {

  fftw_plan fw_plan;
  fftw_plan bw_plan;
  fftw_complex *fftw_data;
  ptrdiff_t global_size_grid; /* Global size of the FFT grid */
  ptrdiff_t local_size_grid;  /* Local size of the FFT grid */
  ptrdiff_t local_n1;         /* Local dimension of n1 */
  ptrdiff_t local_n1_offset;  /* Offset due to possible rests */
  MPI_Comm mpi_comm;

  struct Fft3dInfo *fft_3d_info;

} fftw_mpi_handler;
#define MPI_HANDLER_TYPE 0

#include "../diffusion_fft/utilities.h"
