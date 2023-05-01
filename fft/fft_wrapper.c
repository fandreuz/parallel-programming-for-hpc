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
  fft->fftw_data =
      (fftw_complex *)fftw_malloc(fft->local_size_grid * sizeof(fftw_complex));

  fft->bw_plan =
      fftw_mpi_plan_dft_3d(n1, n2, n3, fft->fftw_data, fft->fftw_data,
                           fft->mpi_comm, FFTW_BACKWARD, FFTW_ESTIMATE);

  fft->fft_3d_info = (struct Fft3dInfo *)malloc(sizeof(struct Fft3dInfo));
  setup_fft3d(fft->fft_3d_info, n1, n2, n3);
}

void close_fftw(fftw_mpi_handler *fft) {
  fftw_destroy_plan(fft->bw_plan);
  fftw_free(fft->fftw_data);

  cleanup_fft3d(fft->fft_3d_info);
  free(fft->fft_3d_info);
}

void fft_3d(fftw_mpi_handler *fft, double *data_direct, fftw_complex *data_rec,
            bool direct_to_reciprocal) {

  double fac;
  int i;

  if (direct_to_reciprocal) {
    fft_3d_2(data_direct, data_rec, fft->fft_3d_info);
  } else {

    memcpy(fft->fftw_data, data_rec,
           fft->local_size_grid * sizeof(fftw_complex));

    fftw_mpi_execute_dft(fft->bw_plan, fft->fftw_data, fft->fftw_data);

    fac = 1.0 / fft->global_size_grid;

    for (i = 0; i < fft->local_size_grid; ++i) {

      data_direct[i] = creal(fft->fftw_data[i]) * fac;
    }
  }
}
