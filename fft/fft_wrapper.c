/* Assignement:
 * Here you have to modify the includes, the array sizes and the fftw calls, to
 * use the fftw-mpi
 *
 * Regarding the fftw calls. here is the substitution
 * fftw_plan_dft_3d -> fftw_mpi_plan_dft_3d
 * ftw_execute_dft  > fftw_mpi_execute_dft
 * use fftw_mpi_local_size_3d for local size of the arrays
 *
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 *
 */

#include "utilities.h"
#include <stdbool.h>
#include <string.h>

void init_fftw(fftw_mpi_handler *fft, int n1, int n2, int n3, MPI_Comm comm) {

  /*
   * Call to fftw_mpi_init is needed to initialize a parallel enviroment for the
   * fftw_mpi
   */

  fftw_mpi_init();
  fft->mpi_comm = comm;

  /*
   *  Allocate a distributed grid for complex FFT using aligned memory
   * allocation See details here:
   *  http://www.fftw.org/fftw3_doc/Allocating-aligned-memory-in-Fortran.html#Allocating-aligned-memory-in-Fortran
   *  HINT: the allocation size is given by fftw_mpi_local_size_3d (see also
   * http://www.fftw.org/doc/MPI-Plan-Creation.html)
   *
   */
  fft->global_size_grid = n1 * n2 * n3;
  fft->local_size_grid = fftw_mpi_local_size_3d(
      n1, n2, n3, fft->mpi_comm, &fft->local_n1, &fft->local_n1_offset);
  fft->fftw_data =
      (fftw_complex *)fftw_malloc(fft->local_size_grid * sizeof(fftw_complex));

  /*
   * Create an FFTW plan for a distributed FFT grid
   * Use fftw_mpi_plan_dft_3d:
   * http://www.fftw.org/doc/MPI-Plan-Creation.html#MPI-Plan-Creation
   */

  fft->fw_plan =
      fftw_mpi_plan_dft_3d(n1, n2, n3, fft->fftw_data, fft->fftw_data,
                           fft->mpi_comm, FFTW_FORWARD, FFTW_ESTIMATE);
  fft->bw_plan =
      fftw_mpi_plan_dft_3d(n1, n2, n3, fft->fftw_data, fft->fftw_data,
                           fft->mpi_comm, FFTW_BACKWARD, FFTW_ESTIMATE);

  fft->fft_3d_info = setup_fft3d(n1, n2, n3);
}

void close_fftw(fftw_mpi_handler *fft) {

  fftw_destroy_plan(fft->bw_plan);
  fftw_destroy_plan(fft->fw_plan);
  fftw_free(fft->fftw_data);

  cleanup_fft3d(fft->fft_3d_info);
}

/* This subroutine uses fftw to calculate 3-dimensional discrete FFTs.
 * The data in direct space is assumed to be real-valued
 * The data in reciprocal space is complex.
 * direct_to_reciprocal indicates in which direction the FFT is to be calculated
 *
 * Note that for real data in direct space (like here), we have
 * F(N-j) = conj(F(j)) where F is the array in reciprocal space.
 * Here, we do not make use of this property.
 * Also, we do not use the special (time-saving) routines of FFTW which
 * allow one to save time and memory for such real-to-complex transforms.
 *
 * f: array in direct space
 * F: array in reciprocal space
 *
 * F(k) = \sum_{l=0}^{N-1} exp(- 2 \pi I k*l/N) f(l)
 * f(l) = 1/N \sum_{k=0}^{N-1} exp(+ 2 \pi I k*l/N) F(k)
 *
 */

void fft_3d(fftw_mpi_handler *fft, double *data_direct, fftw_complex *data_rec,
            bool direct_to_reciprocal) {

  double fac;
  int i;

  // Now distinguish in which direction the FFT is performed
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
