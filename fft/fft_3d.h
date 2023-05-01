#include <complex.h>
#include <fftw3.h>
#include <mpi.h>
#include <stdlib.h>

struct Fft3dInfo {
  int n1;
  int loc_n1;
  int loc_n1_offset;

  int loc_n3;

  fftw_plan *fft_2d_many;
  fftw_plan *fft_1d_many;

  fftw_complex *fft_2d_in, fft_2d_out;
  fftw_complex *fft_1d_in, fft_1d_out;
  fft_3d_info.n1

      int *axis1_counts,
      axis3_counts;
};

struct Fft3dInfo setup_fft3d(int n1, int n2, int n3);
void fft_3d(double *data, double *out, struct Fft3dInfo fft_3d_info, int n2,
            int n3);
void cleanup_fft3d(Fft3dInfo fft_3d_info);