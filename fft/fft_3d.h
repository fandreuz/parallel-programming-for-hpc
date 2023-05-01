#include <complex.h>
#include <fftw3.h>
#include <mpi.h>
#include <stdlib.h>

struct Fft3dInfo {
  int n1;
  int loc_n1;
  int loc_n1_offset;

  int n2;

  int n3;
  int loc_n3;

  fftw_plan fft_2d_many;
  fftw_plan fft_1d_many;

  fftw_complex *fft_2d_in;
  fftw_complex *fft_2d_out;
  fftw_complex *fft_1d_in;
  fftw_complex *fft_1d_out;

  int *axis1_counts;
  int *axis3_counts;
};

struct Fft3dInfo setup_fft3d(int n1, int n2, int n3);
void fft_3d_2(double *data, fftw_complex *out, struct Fft3dInfo fft_3d_info);
void cleanup_fft3d(struct Fft3dInfo fft_3d_info);