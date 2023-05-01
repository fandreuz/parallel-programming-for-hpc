/*
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 */

#ifndef _FFTW_UTLITIES_
#define _FFTW_UTLITIES_
#include <complex.h>
#include <fftw3-mpi.h>
#include <fftw3.h>
#include <mpi.h>
#include <stdbool.h>
#include <sys/time.h>
#define pi 3.14159265358979323846

#ifndef MPI_HANDLER_TYPE

typedef struct {

  fftw_plan fw_plan;
  fftw_plan bw_plan;
  fftw_complex *fftw_data;
  ptrdiff_t global_size_grid; /* Global size of the FFT grid */
  ptrdiff_t local_size_grid;  /* Local size of the FFT grid */
  ptrdiff_t local_n1;         /* Local dimension of n1 */
  ptrdiff_t local_n1_offset;  /* Offset due to possible rests */
  MPI_Comm mpi_comm;

} fftw_mpi_handler;

#endif MPI_HANDLER_TYPE

inline double seconds() {

  /*
   * Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)
   *
   */
  struct timeval tmp;
  double sec;

  gettimeofday(&tmp, (struct timezone *)0);
  sec = tmp.tv_sec + ((double)tmp.tv_usec) / 1000000.0;

  return sec;
}

/*
 * Index linearization is computed following row-major order.
 * For more informtion see FFTW documentation:
 * http://www.fftw.org/doc/Row_002dmajor-Format.html#Row_002dmajor-Format
 *
 */
inline int index_f(int i1, int i2, int i3, int n1, int n2, int n3) {
  return n3 * n2 * i1 + n3 * i2 + i3;
}

void plot_data_1d(char *name, int n1, int n2, int n3, int n1_local,
                  int n1_local_offset, int dir, double *data);
void plot_data_2d(char *name, int n1, int n2, int n3, int n1_local,
                  int n1_local_offset, int dir, double *data);
void init_fftw(fftw_mpi_handler *fft, int n1, int n2, int n3,
               MPI_Comm mpi_comm);
void close_fftw(fftw_mpi_handler *fft);

void derivative(fftw_mpi_handler *fft, int n1, int n2, int n3, double L1,
                double L2, double L3, int ipol, double *data, double *deriv);

/* New interface for fft_3d which includes a parameter of kind fftw_mpi_handler
 */
void fft_3d(fftw_mpi_handler *fft, double *data_direct, fftw_complex *data_rec,
            bool direct_to_reciprocal);

#endif
