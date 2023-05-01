#pragma once

#include "fft_3d.h"
#include <fftw3.h>
#include <mpi.h>

typedef struct {

  fftw_plan fw_plan;
  fftw_plan bw_plan;
  fftw_complex *fftw_data;
  ptrdiff_t global_size_grid; /* Global size of the FFT grid */
  ptrdiff_t local_size_grid;  /* Local size of the FFT grid */
  ptrdiff_t local_n1;         /* Local dimension of n1 */
  ptrdiff_t local_n1_offset;  /* Offset due to possible rests */
  MPI_Comm mpi_comm;

  struct Fft3dInfo fft_3d_info;

} fftw_mpi_handler;
#define MPI_HANDLER_TYPE 0

#include "../diffusion_fft/src/utilities.h"
