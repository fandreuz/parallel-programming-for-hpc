#include "fft_3d.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void setup_fft3d(struct Fft3dInfo *info, int n1, int n2, int n3) {
  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int div = n1 / nProcesses;
  int res = n1 % nProcesses;

  info->n1 = n1;
  info->loc_n1 = div;
  info->loc_n1 += locRank < res;
  info->loc_n1_offset = div * locRank + MIN(locRank, res);

  info->n2 = n2;
  info->n3 = n3;
  info->loc_n3 = n3 / nProcesses;
  info->loc_n3 += locRank < (n3 % nProcesses);

  info->axis1_counts = (int *)malloc(sizeof(int) * nProcesses);
  int div_n1 = n1 / nProcesses;
  for (int i = 0; i < n3 % nProcesses; ++i) {
    info->axis1_counts[i] = div_n1 + 1;
  }
  for (int i = n1 % nProcesses; i < nProcesses; ++i) {
    info->axis1_counts[i] = div_n1;
  }

  info->axis3_counts = (int *)malloc(sizeof(int) * nProcesses);
  int div_n3 = n3 / nProcesses;
  for (int i = 0; i < n3 % nProcesses; ++i) {
    info->axis3_counts[i] = div_n3 + 1;
  }
  for (int i = n3 % nProcesses; i < nProcesses; ++i) {
    info->axis3_counts[i] = div_n3;
  }

  int oldBlockSize = info->loc_n1 * n2 * n3;
  info->fft_2d_in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * oldBlockSize);
  info->fft_2d_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * oldBlockSize);

  int fft_plan_size[] = {n2, n3};
  info->fft_2d_many = fftw_plan_many_dft(
      2, fft_plan_size, info->loc_n1, info->fft_2d_in, fft_plan_size, 1,
      fft_plan_size[0] * fft_plan_size[1], info->fft_2d_out, fft_plan_size, 1,
      fft_plan_size[0] * fft_plan_size[1], FFTW_FORWARD, FFTW_ESTIMATE);

  int newBlockSize = info->loc_n3 * n2 * info->n1;
  info->fft_1d_in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * newBlockSize);
  info->fft_1d_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * newBlockSize);

  int fft2_plan_size[] = {n1};
  info->fft_1d_many = fftw_plan_many_dft(
      1, fft2_plan_size, n2 * info->loc_n3, info->fft_1d_in, fft2_plan_size, 1,
      fft2_plan_size[0], info->fft_1d_out, fft2_plan_size, 1, fft2_plan_size[0],
      FFTW_FORWARD, FFTW_ESTIMATE);
}

void swap_1_3(fftw_complex *data, fftw_complex *out, int n1, int n2, int n3) {
  for (int i1 = 0; i1 < n1; ++i1) {
    for (int i2 = 0; i2 < n2; ++i2) {
      for (int i3 = 0; i3 < n3; ++i3) {
        out[i3 * n2 * n1 + i2 * n1 + i1] = data[i1 * n2 * n3 + i2 * n3 + i3];
      }
    }
  }
}

void send_split(fftw_complex *data, fftw_complex *out, int n2,
                int *axis1_counts, int *axis3_counts) {
  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int *send_counts = (int *)malloc(sizeof(int) * nProcesses);
  int *send_displacements = (int *)malloc(sizeof(int) * nProcesses);

  int slice_size = axis1_counts[locRank] * n2;
  send_counts[0] = axis3_counts[0] * slice_size;
  send_displacements[0] = 0;
  for (int i = 0; i < nProcesses; ++i) {
    send_counts[i] = axis3_counts[i] * slice_size;
    send_displacements[i] = send_counts[i] + send_displacements[i - 1];
  }

  int *recv_counts = (int *)malloc(sizeof(int) * nProcesses);
  int *recv_displacements = (int *)malloc(sizeof(int) * nProcesses);

  recv_counts[0] = axis1_counts[0] * axis3_counts[0] * n2;
  recv_displacements[0] = 0;
  for (int i = 1; i < nProcesses; ++i) {
    recv_counts[i] = axis1_counts[i] * axis3_counts[i] * n2;
    recv_displacements[i] = recv_counts[i] + recv_displacements[i - 1];
  }

  MPI_Datatype fftw_complex_mpi;
  MPI_Type_contiguous(2, MPI_DOUBLE, &fftw_complex_mpi);
  MPI_Type_commit(&fftw_complex_mpi);
  
  MPI_Alltoallv(data, send_counts, send_displacements, fftw_complex_mpi, out,
                recv_counts, recv_displacements, fftw_complex_mpi,
                MPI_COMM_WORLD);
}

void fft_3d_2(double *data, fftw_complex *out, struct Fft3dInfo *fft_3d_info) {
  for (int i = 0; i < fft_3d_info->loc_n1 * fft_3d_info->n2 * fft_3d_info->n3;
       i++) {
    fft_3d_info->fft_2d_in[i] = data[i] + 0.0 * I;
  }
  fftw_execute(fft_3d_info->fft_2d_many);
  // swap fft_2d_out into fft_2d_in
  swap_1_3(fft_3d_info->fft_2d_out, fft_3d_info->fft_2d_in, fft_3d_info->loc_n1,
           fft_3d_info->n2, fft_3d_info->n3);

  send_split(fft_3d_info->fft_2d_in, fft_3d_info->fft_1d_in, fft_3d_info->n2,
             fft_3d_info->axis1_counts, fft_3d_info->axis3_counts);
  fftw_execute(fft_3d_info->fft_1d_many);

  swap_1_3(fft_3d_info->fft_1d_out, fft_3d_info->fft_1d_in, fft_3d_info->loc_n3,
           fft_3d_info->n2, fft_3d_info->n1);

  send_split(fft_3d_info->fft_1d_in, out, fft_3d_info->n2,
             fft_3d_info->axis3_counts, fft_3d_info->axis1_counts);
}

void cleanup_fft3d(struct Fft3dInfo *fft_3d_info) {
  fftw_destroy_plan(fft_3d_info->fft_2d_many);
  fftw_destroy_plan(fft_3d_info->fft_1d_many);

  fftw_free(fft_3d_info->fft_2d_in);
  fftw_free(fft_3d_info->fft_2d_out);
  fftw_free(fft_3d_info->fft_1d_in);
  fftw_free(fft_3d_info->fft_1d_out);
}