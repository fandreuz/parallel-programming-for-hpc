#include "fft_3d.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

DataInfo setup_fft_3d(int n1) {
  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int div = n1 / nProcesses;
  int res = n1 % nProcesses;

  DataInfo info;
  info.loc_n1 = div + locRank < res;
  info.loc_n1_offset = div * locRank + MIN(locRank, res);
  return info;
}

void *swap_1_3(fftw_complex *data, fftw_complex *out, int n1, int n2, int n3) {
  for (int i1 = 0; i1 < n1; ++i1) {
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i3 = 0; i3 < n3; ++i3) {
        out[i3 * n2 * n1 + i2 * n1 + i1] = data[i1 * n2 * n3 + i2 * n3 + i3];
      }
    }
  }
}

void send_split(fftw_complex *data, fftw_complex *out, int n1, int n2, int n3) {
  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int div_n1 = n1 / nProcesses;
  for (int i = 0; i < n3 % nProcesses; ++i) {
    axis1_counts[i] = div_n1 + 1;
  }
  for (int i = n1 % nProcesses; i < nProcesses; ++i) {
    axis1_counts[i] = div_n1;
  }

  int div_n3 = n3 / nProcesses;
  for (int i = 0; i < n3 % nProcesses; ++i) {
    axis3_counts[i] = div_n3 + 1;
  }
  for (int i = n3 % nProcesses; i < nProcesses; ++i) {
    axis3_counts[i] = div_n3;
  }

  int *axis3_counts = (int *)malloc(sizeof(int) nProcesses);
  int *send_counts = (int *)malloc(sizeof(int) nProcesses);
  int *send_displacements = (int *)malloc(sizeof(int) nProcesses);

  int slice_size = axis1_counts[locRank] * n2;
  send_counts[0] = axis3_counts[0] * slice_size;
  send_displacements[0] = 0;
  for (int i = 0; i < nProcesses; ++i) {
    send_counts[i] = axis3_counts[i] * slice_size;
    send_displacements[i] = send_counts[i] + send_displacements[i - 1];
  }

  int *axis1_counts = (int *)malloc(sizeof(int) nProcesses);
  int *recv_counts = (int *)malloc(sizeof(int) nProcesses);
  int *recv_displacements = (int *)malloc(sizeof(int) nProcesses);

  recv_counts[0] = axis1_counts[0] * axis3_counts[0] * n2;
  recv_displacements[0] = 0;
  for (int i = 1; i < nProcesses; ++i) {
    recv_counts[i] = axis1_counts[i] * axis3_counts[i] * n2;
    recv_displacements[i] = recv_counts[i] + recv_displacements[i - 1];
  }

  MPI_Alltoallv(data, send_counts, send_displacements, MPI_C_DOUBLE_COMPLEX,
                recv_buffer, recv_counts, recv_displacements,
                MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
}

fftw_complex *fft_3d(double *data, int loc_n1, int n2, int n3) {
  int N = loc_n1 * n2 * n3;
  fftw_complex *fft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
  fftw_complex *fft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

  int fft_plan_size[] = {n2, n3};
  fftw_plan many_dft_plan = fftw_plan_many_dft(
      2, fft_plan_size, loc_n1, in, NULL, 1,
      fft_plan_size[0] * fft_plan_size[1], out, NULL, 1,
      fft_plan_size[0] * fft_plan_size[1], FFTW_FORWARD, FFTW_ESTIMATE);

  for (int i = 0; i < loc_n1 * n2 * n3; i++) {
    fft_in[i] = data[i] + 0.0 * I;
  }

  fftw_execute(many_dft_plan);

  fftw_complex *fft_out_swap =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * loc_n1 * n2 * n3);
  swap_1_3(fft_out, fft_out_swap, loc_n1, n2, n3);

  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
  int newLocN3 = n3 / nProcesses + n3 < (n3 % nProcesses);
  int newBlockSize = newLocN3 * n2 * n1;

  // axis3 - axis 2 - axis 1
  fftw_complex *exchanged =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * newBlockSize);
  send_split(fft_out_swap, exchanged, n1, n2, n3);

  fftw_complex *fft_out2 =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * newBlockSize);

  int fft2_plan_size = {n1};
  fftw_plan many_dft_plan2 = fftw_plan_many_dft(
      1, fft2_plan_size, n2 * newLocN3, exchanged, NULL, 1, fft2_plan_size[0],
      fft_out2, NULL, 1, fft2_plan_size[0], FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(many_dft_plan2);

  swap_1_3(fft_out2, exchanged, newLocN3, n2, n1);
  send_split(exchanged, fft_out, n1, n2, n3);

  return fft_out;
}