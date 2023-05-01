#include <complex.h>
#include <fftw3.h>
#include <mpi.h>

struct DataInfo {
    int loc_n1;
    int loc_n1_offset;
};

struct DataInfo setup_fft_3d(int n1);
fftw_complex * fft_3d(double *data, int loc_n1, int n2, int n3);