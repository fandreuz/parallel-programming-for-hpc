#include "fft_3d.h"
#include "utilities.h"
#include <stdbool.h>
#include <string.h>

void init_fftw(fftw_mpi_handler *fft, int n1, int n2, int n3, MPI_Comm comm) {
  fftw_mpi_init();
  fft->mpi_comm = comm;

  fft->global_size_grid = n1 * n2 * n3;
  fft->local_size_grid = fftw_mpi_local_size_3d(
      n1, n2, n3, fft->mpi_comm, &fft->local_n1, &fft->local_n1_offset);

  fft->fft_3d_info = (struct Fft3dInfo *)malloc(sizeof(struct Fft3dInfo));
  setup_fft3d(fft->fft_3d_info, n1, n2, n3);
}

void close_fftw(fftw_mpi_handler *fft) {
  cleanup_fft3d(fft->fft_3d_info);
  free(fft->fft_3d_info);
}

void fft_3d(fftw_mpi_handler *fft, double *data_direct, fftw_complex *data_rec,
            bool direct_to_reciprocal) {
  if (direct_to_reciprocal) {
    fft_3d_2(data_direct, data_rec, fft->fft_3d_info);
  } else {
    ifft_3d_2(data_rec, data_direct, fft->fft_3d_info);
  }
}
