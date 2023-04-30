/* Assignment:
 * Parallelize the code, using fftw-mpi
 * This amount to
 *   - distribute the data contained in diffusivity, conc, dconc, aux1, aux2 in
 * the way fftw-mpi expects
 *   - modify the fftw calls in fftw-mpi in p_fftw_wrapper
 * You will need to modify the files
 *   - diffusion.c
 *   - derivative.c
 *   - fftw_wrapper.c
 * In these files you will find some HINTs, make good use of them :)
 *
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 */

#include "utilities.h"
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
// HINT: include mpi and fftw3-mpi
//       http://www.fftw.org/doc/MPI-Files-and-Data-Types.html#MPI-Files-and-Data-Types

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  // Dimensions of the system
  const double L1 = 10., L2 = 10., L3 = 20.;
  // Grid size
  const int n1 = 48, n2 = 48, n3 = 96;
  // time step for time integration
  const double dt = 2.e-3;
  // number of time steps
  const int nstep = 101;
  // Radius of diffusion channel
  const double rad_diff = 0.7;
  // Radius of starting concentration
  const double rad_conc = 0.6;

  /*
   * initialize the fftw system and local dimension
   * as the value returned from the parallel FFT grid initializzation
   */
  fftw_mpi_handler fft_h;
  init_fftw(&fft_h, n1, n2, n3, MPI_COMM_WORLD);

  double *diffusivity =
      (double *)malloc(fft->local_n1 * n2 * n3 * sizeof(double));
  double *conc = (double *)malloc(fft->local_n1 * n2 * n3 * sizeof(double));
  double *dconc = (double *)malloc(fft->local_n1 * n2 * n3 * sizeof(double));

  /*
   * Define the diffusivity inside the system and
   * the starting concentration
   */
  double ss = 0.0;

  for (int i3 = 0; i3 < n3; ++i3) {
    double x3 = L3 * ((double)i3) / n3;
    double f3diff = exp(-pow((x3 - 0.5 * L3) / rad_diff, 2));
    double f3conc = exp(-pow((x3 - 0.5 * L3) / rad_conc, 2));

    for (int i2 = 0; i2 < n2; ++i2) {
      double x2 = L2 * ((double)i2) / n2;
      double f2diff = exp(-pow((x2 - 0.5 * L2) / rad_diff, 2));
      double f2conc = exp(-pow((x2 - 0.5 * L2) / rad_conc, 2));

      for (int i1 = 0; i1 < fft->local_n1; ++i1) {
        double x1 = L1 * ((double)(i1 + fft->local_n1_offset)) / n1;
        double f1diff = exp(-pow((x1 - 0.5 * L1) / rad_diff, 2));
        double f1conc = exp(-pow((x1 - 0.5 * L1) / rad_conc, 2));

        int index = index_f(i1, i2, i3, fft->local_n1, n2, n3);
        diffusivity[index] = MAX(f1diff * f2diff, f2diff * f3diff);
        conc[index] = f1conc * f2conc * f3conc;
        ss += conc[index];
      }
    }
  }

  MPI_Allreduce(&ss, &ss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /*
   * HINT: The parallel version of  the output routines is provided in the
   * mpi_output_routines folder
   *
   */
  plot_data_2d("diffusivity", n1, n2, n3, fft->local_n1, fft->local_n1_offset,
               1, diffusivity);
  plot_data_2d("diffusivity", n1, n2, n3, fft->local_n1, fft->local_n1_offset,
               2, diffusivity);
  plot_data_2d("diffusivity", n1, n2, n3, fft->local_n1, fft->local_n1_offset,
               3, diffusivity);

  double fac = L1 * L2 * L3 / (n1 * n2 * n3);

  /*
   * Normalize the concentration.
   */
  ss = 1.0 / (ss * fac);
  for (int i1 = 0; i1 < fft->local_n1 * n2 * n3; ++i1)
    conc[i1] *= ss;

  double *aux1 = (double *)malloc(fft->local_n1 * n2 * n3 * sizeof(double));
  double *aux2 = (double *)malloc(fft->local_n1 * n2 * n3 * sizeof(double));
  double *buffer = (double *)malloc(2 * sizeof(double));

  /*
   * Now everything is defined: system size, diffusivity inside the system, and
   * the starting concentration
   *
   * Start the dynamics
   *
   */
  double start = seconds();
  for (int istep = 1; istep <= nstep; ++istep) {
    for (int i1 = 0; i1 < fft->local_n1 * n2 * n3; ++i1)
      dconc[i1] = 0.0;

    for (int ipol = 1; ipol <= 3; ++ipol) {
      derivative(&fft_h, n1, n2, n3, L1, L2, L3, ipol, conc, aux1);
      for (int i1 = 0; i1 < fft->local_n1 * n2 * n3; ++i1) {
        aux1[i1] *= diffusivity[i1];
      }

      derivative(&fft_h, n1, n2, n3, L1, L2, L3, ipol, aux1, aux2);
      // summing up contributions from the three spatial directions
      for (int i1 = 0; i1 < fft->local_n1 * n2 * n3; ++i1)
        dconc[i1] += aux2[i1];
    }

    for (int i1 = 0; i1 < fft->local_n1 * n2 * n3; ++i1)
      conc[i1] += dt * dconc[i1];

    if (istep % 30 == 1) {
      // Check the normalization of conc
      buffer[0] = 0.0; // ss
      buffer[1] = 0.0; // r2mean

      for (int i3 = 0; i3 < n3; ++i3) {
        double x3 = L3 * ((double)i3) / n3 - 0.5 * L3;
        for (int i2 = 0; i2 < n2; ++i2) {
          double x2 = L2 * ((double)i2) / n2 - 0.5 * L2;
          for (int i1 = 0; i1 < fft->local_n1; ++i1) {
            double x1 = L1 * ((double)i1) / n1 - 0.5 * L1;
            double rr = pow(x1, 2) + pow(x2, 2) + pow(x3, 2);
            int index = index_f(i1, i2, i3, fft->local_n1, n2, n3);

            buffer[0] += conc[index];
            buffer[1] += conc[index] * rr;
          }
        }
      }

      MPI_Allreduce(&buffer, &buffer, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      buffer[0] *= fac;
      buffer[1] *= fac;

      double end = seconds();
      printf(" %d %17.15f %17.15f Elapsed time per iteration %f \n ", istep,
             buffer[1], buffer[0], (end - start) / istep);

      plot_data_2d("concentration", n1, n2, n3, fft->local_n1,
                   fft->local_n1_offset, 2, conc);
      plot_data_1d("1d_conc", n1, n2, n3, fft->local_n1, fft->local_n1_offset,
                   3, conc);
    }
  }

  close_fftw(&fft_h);
  free(buffer);
  free(diffusivity);
  free(conc);
  free(dconc);
  free(aux1);
  free(aux2);

  MPI_Finalize();

  return 0;
}
