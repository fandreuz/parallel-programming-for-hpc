#include "fft_3d.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void setup_fft3d(struct Fft3dInfo *info, int n1, int n2, int n3) {
  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int div = n1 / nProcesses;
  int res = n1 % nProcesses;

  info->nProcesses = nProcesses;

  info->n1 = n1;
  info->loc_n1 = div;
  info->loc_n1 += locRank < res;
  info->loc_n1_offset = div * locRank + MIN(locRank, res);

  info->n2 = n2;
  info->n3 = n3;
  info->loc_n3 = n3 / nProcesses;
  info->loc_n3 += locRank < (n3 % nProcesses);

  // compute portion of axis for each processor, on axis 1 and 3

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

  // compute send/recv_counts/displacements

  info->send_counts = (int *)malloc(sizeof(int) * nProcesses);
  info->send_displacements = (int *)malloc(sizeof(int) * nProcesses);

  int slice_size = info->axis1_counts[locRank] * n2;
  info->send_counts[0] = info->axis3_counts[0] * slice_size;
  info->send_displacements[0] = 0;
  for (int i = 1; i < nProcesses; ++i) {
    info->send_counts[i] = info->axis3_counts[i] * slice_size;
    info->send_displacements[i] =
        info->send_counts[i - 1] + info->send_displacements[i - 1];
  }

  info->recv_counts = (int *)malloc(sizeof(int) * nProcesses);
  info->recv_displacements = (int *)malloc(sizeof(int) * nProcesses);

  int recv_slice_size = info->axis3_counts[locRank] * n2;
  info->recv_counts[0] = info->axis1_counts[0] * recv_slice_size;
  info->recv_displacements[0] = 0;
  for (int i = 1; i < nProcesses; ++i) {
    info->recv_counts[i] = info->axis1_counts[i] * recv_slice_size;
    info->recv_displacements[i] =
        info->recv_counts[i - 1] + info->recv_displacements[i - 1];
  }

  // initialize FFTW in/out buffers

  int blockSize2d = info->loc_n1 * n2 * n3;
  info->fft_2d_in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * blockSize2d);
  info->fft_2d_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * blockSize2d);

  int blockSize1d = info->loc_n3 * n2 * n1;
  info->fft_1d_in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * blockSize1d);
  info->fft_1d_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * blockSize1d);

  // initialize FFTW plans

  int fft_plan_size[] = {n2, n3};
  info->fft_2d_many = fftw_plan_many_dft(
      2, fft_plan_size, info->loc_n1, info->fft_2d_in, fft_plan_size, 1,
      fft_plan_size[0] * fft_plan_size[1], info->fft_2d_out, fft_plan_size, 1,
      fft_plan_size[0] * fft_plan_size[1], FFTW_FORWARD, FFTW_ESTIMATE);
  info->ifft_2d_many = fftw_plan_many_dft(
      2, fft_plan_size, info->loc_n1, info->fft_2d_in, fft_plan_size, 1,
      fft_plan_size[0] * fft_plan_size[1], info->fft_2d_out, fft_plan_size, 1,
      fft_plan_size[0] * fft_plan_size[1], FFTW_BACKWARD, FFTW_ESTIMATE);

  int fft2_plan_size[] = {n1};
  info->fft_1d_many = fftw_plan_many_dft(
      1, fft2_plan_size, n2 * info->loc_n3, info->fft_1d_in, fft2_plan_size, 1,
      fft2_plan_size[0], info->fft_1d_out, fft2_plan_size, 1, fft2_plan_size[0],
      FFTW_FORWARD, FFTW_ESTIMATE);
  info->ifft_1d_many = fftw_plan_many_dft(
      1, fft2_plan_size, n2 * info->loc_n3, info->fft_1d_in, fft2_plan_size, 1,
      fft2_plan_size[0], info->fft_1d_out, fft2_plan_size, 1, fft2_plan_size[0],
      FFTW_BACKWARD, FFTW_ESTIMATE);
}

void rectify_3(fftw_complex *data, fftw_complex *out, int n1, int n2, int n3,
               int *n3_counts, int nProcesses) {
  int i3_start = 0;
  fftw_complex *data_walking = data;
  for (int proc = 0; proc < nProcesses; ++proc) {
    int i3_count = n3_counts[proc];

    for (int i1 = 0; i1 < n1; ++i1) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i3 = i3_start; i3 < i3_start + i3_count; ++i3) {
          out[i3 + i2 * n3 + i1 * n3 * n2] = *(data_walking++);
        }
      }
    }

    i3_start += i3_count;
  }
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

void send_split(fftw_complex *data, fftw_complex *out,
                struct Fft3dInfo *fft_3d_info) {
  MPI_Alltoallv(data, fft_3d_info->send_counts, fft_3d_info->send_displacements,
                MPI_C_DOUBLE_COMPLEX, out, fft_3d_info->recv_counts,
                fft_3d_info->recv_displacements, MPI_C_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);
}

void send_split_back(fftw_complex *data, fftw_complex *out,
                     struct Fft3dInfo *fft_3d_info) {
  MPI_Alltoallv(data, fft_3d_info->recv_counts, fft_3d_info->recv_displacements,
                MPI_C_DOUBLE_COMPLEX, out, fft_3d_info->send_counts,
                fft_3d_info->send_displacements, MPI_C_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);
}

void fft_3d_2(double *data, fftw_complex *out, struct Fft3dInfo *fft_3d_info) {
  for (int i = 0; i < fft_3d_info->loc_n1 * fft_3d_info->n2 * fft_3d_info->n3;
       i++) {
    fft_3d_info->fft_2d_in[i] = data[i] + 0.0 * I;
  }
  fftw_execute(fft_3d_info->fft_2d_many);

  swap_1_3(fft_3d_info->fft_2d_out, fft_3d_info->fft_2d_in, fft_3d_info->loc_n1,
           fft_3d_info->n2, fft_3d_info->n3);
  send_split(fft_3d_info->fft_2d_in, fft_3d_info->fft_1d_out, fft_3d_info);
  rectify_3(fft_3d_info->fft_1d_out, fft_3d_info->fft_1d_in,
            fft_3d_info->loc_n3, fft_3d_info->n2, fft_3d_info->n1,
            fft_3d_info->axis1_counts, fft_3d_info->nProcesses);

  fftw_execute(fft_3d_info->fft_1d_many);

  swap_1_3(fft_3d_info->fft_1d_out, fft_3d_info->fft_1d_in, fft_3d_info->loc_n3,
           fft_3d_info->n2, fft_3d_info->n1);
  send_split_back(fft_3d_info->fft_1d_in, fft_3d_info->fft_2d_in, fft_3d_info);
  rectify_3(fft_3d_info->fft_2d_in, out, fft_3d_info->loc_n1, fft_3d_info->n2,
            fft_3d_info->n3, fft_3d_info->axis3_counts,
            fft_3d_info->nProcesses);
}

void ifft_3d_2(fftw_complex *data, double *out, struct Fft3dInfo *fft_3d_info) {
  int loc_grid_size = fft_3d_info->loc_n1 * fft_3d_info->n2 * fft_3d_info->n3;
  memcpy(fft_3d_info->fft_2d_in, data, loc_grid_size * sizeof(fftw_complex));
  fftw_execute(fft_3d_info->ifft_2d_many);

  swap_1_3(fft_3d_info->fft_2d_out, fft_3d_info->fft_2d_in, fft_3d_info->loc_n1,
           fft_3d_info->n2, fft_3d_info->n3);
  send_split(fft_3d_info->fft_2d_in, fft_3d_info->fft_1d_out, fft_3d_info);
  rectify_3(fft_3d_info->fft_1d_out, fft_3d_info->fft_1d_in,
            fft_3d_info->loc_n3, fft_3d_info->n2, fft_3d_info->n1,
            fft_3d_info->axis1_counts, fft_3d_info->nProcesses);

  fftw_execute(fft_3d_info->ifft_1d_many);

  swap_1_3(fft_3d_info->fft_1d_out, fft_3d_info->fft_1d_in, fft_3d_info->loc_n3,
           fft_3d_info->n2, fft_3d_info->n1);
  send_split_back(fft_3d_info->fft_1d_in, fft_3d_info->fft_2d_in, fft_3d_info);
  rectify_3(fft_3d_info->fft_2d_in, fft_3d_info->fft_2d_out,
            fft_3d_info->loc_n1, fft_3d_info->n2, fft_3d_info->n3,
            fft_3d_info->axis3_counts, fft_3d_info->nProcesses);

  double fac = 1.0 / (fft_3d_info->n1 * fft_3d_info->n2 * fft_3d_info->n3);
  for (int i = 0; i < loc_grid_size; ++i) {
    out[i] = creal(fft_3d_info->fft_2d_out[i]) * fac;
  }
}

void cleanup_fft3d(struct Fft3dInfo *fft_3d_info) {
  fftw_destroy_plan(fft_3d_info->fft_2d_many);
  fftw_destroy_plan(fft_3d_info->fft_1d_many);
  fftw_destroy_plan(fft_3d_info->ifft_2d_many);
  fftw_destroy_plan(fft_3d_info->ifft_1d_many);

  fftw_free(fft_3d_info->fft_2d_in);
  fftw_free(fft_3d_info->fft_2d_out);
  fftw_free(fft_3d_info->fft_1d_in);
  fftw_free(fft_3d_info->fft_1d_out);
}