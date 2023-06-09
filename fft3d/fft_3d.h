#include "utilities.h"

void setup_fft3d(struct Fft3dInfo *info, int n1, int n2, int n3);
void fft_3d_2(double *data, fftw_complex *out, struct Fft3dInfo *fft_3d_info);
void ifft_3d_2(fftw_complex *data, double *out, struct Fft3dInfo *fft_3d_info);
void cleanup_fft3d(struct Fft3dInfo *fft_3d_info);