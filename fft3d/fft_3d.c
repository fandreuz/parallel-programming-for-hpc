#include "fft_3d.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void setup_fft3d(struct Fft3dInfo *info, int n1, int n2, int n3) {
  int locRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int div_n1 = n1 / nProcesses;
  int res_n1 = n1 % nProcesses;
  info->loc_n1_offset = div_n1 * locRank + MIN(locRank, res_n1);

  int div_n2 = n2 / nProcesses;

  info->nProcesses = nProcesses;

  info->n1 = n1;
  info->n2 = n2;
  info->n3 = n3;

  // compute portion of axis for each processor, on axis 1 and 3

  info->axis1_counts = (int *)malloc(sizeof(int) * nProcesses);
  for (int i = 0; i < n1 % nProcesses; ++i) {
    info->axis1_counts[i] = div_n1 + 1;
  }
  for (int i = n1 % nProcesses; i < nProcesses; ++i) {
    info->axis1_counts[i] = div_n1;
  }

  info->axis2_counts = (int *)malloc(sizeof(int) * nProcesses);
  for (int i = 0; i < n2 % nProcesses; ++i) {
    info->axis2_counts[i] = div_n2 + 1;
  }
  for (int i = n2 % nProcesses; i < nProcesses; ++i) {
    info->axis2_counts[i] = div_n2;
  }

  info->loc_n1 = info->axis1_counts[locRank];
  info->loc_n2 = info->axis2_counts[locRank];

  // compute send/recv_counts/displacements

  info->send_counts = (int *)malloc(sizeof(int) * nProcesses);
  info->send_displacements = (int *)malloc(sizeof(int) * nProcesses);

  int slice_size = info->loc_n1 * n3;
  info->send_counts[0] = info->axis2_counts[0] * slice_size;
  info->send_displacements[0] = 0;
  for (int i = 1; i < nProcesses; ++i) {
    info->send_counts[i] = info->axis2_counts[i] * slice_size;
    info->send_displacements[i] =
        info->send_counts[i - 1] + info->send_displacements[i - 1];
  }

  info->recv_counts = (int *)malloc(sizeof(int) * nProcesses);
  info->recv_displacements = (int *)malloc(sizeof(int) * nProcesses);

  int recv_slice_size = info->loc_n2 * n3;
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

  int blockSize1d = info->loc_n2 * n1 * n3;
  info->fft_1d_in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * blockSize1d);
  info->fft_1d_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * blockSize1d);

  // initialize FFTW plans

  int fft_plan_size[] = {n2, n3};
  info->fft_2d_many = fftw_plan_many_dft(  //
      2,                                   // rank
      fft_plan_size,                       // *n
      info->loc_n1,                        // howmany
      info->fft_2d_in,                     // *in
      fft_plan_size,                       // inembed
      1,                                   // istride
      fft_plan_size[0] * fft_plan_size[1], // idist
      info->fft_2d_out,                    // *out
      fft_plan_size,                       // *onembed
      1,                                   // ostride
      fft_plan_size[0] * fft_plan_size[1], // odist
      FFTW_FORWARD,                        // sign
      FFTW_ESTIMATE                        // flags
  );
  info->ifft_2d_many = fftw_plan_many_dft( //
      2,                                   // rank
      fft_plan_size,                       // *n
      info->loc_n1,                        // howmany
      info->fft_2d_in,                     // *in
      fft_plan_size,                       // inembed
      1,                                   // istride
      fft_plan_size[0] * fft_plan_size[1], // idist
      info->fft_2d_out,                    // *out
      fft_plan_size,                       // *onembed
      1,                                   // ostride
      fft_plan_size[0] * fft_plan_size[1], // odist
      FFTW_BACKWARD,                       // sign
      FFTW_ESTIMATE                        // flags
  );

  int fft2_plan_size[] = {n1};
  info->fft_1d_many = fftw_plan_many_dft( //
      1,                                  // rank
      fft2_plan_size,                     // *n
      recv_slice_size,                    // howmany
      info->fft_1d_in,                    // *in
      fft2_plan_size,                     // inembed
      recv_slice_size,                    // istride
      1,                                  // idist
      info->fft_1d_out,                   // *out
      fft2_plan_size,                     // *onembed
      recv_slice_size,                    // ostride
      1,                                  // odist
      FFTW_FORWARD,                       // sign
      FFTW_ESTIMATE                       // flags
  );
  info->ifft_1d_many = fftw_plan_many_dft( //
      1,                                   // rank
      fft2_plan_size,                      // *n
      recv_slice_size,                     // howmany
      info->fft_1d_in,                     // *in
      fft2_plan_size,                      // inembed
      recv_slice_size,                     // istride
      1,                                   // idist
      info->fft_1d_out,                    // *out
      fft2_plan_size,                      // *onembed
      recv_slice_size,                     // ostride
      1,                                   // odist
      FFTW_BACKWARD,                       // sign
      FFTW_ESTIMATE                        // flags
  );
}

/**
 * Move data such that the order of axes is preserved, but introducing a
 * partition along the second axis, i.e. an entire slice along the first
 * axis must be visited before moving to the next partition.
 */
void partition_axis_2(fftw_complex *data, fftw_complex *out, int n1, int n2,
                      int n3, int nprocesses, int *partition_sizes) {
  int current_n2_offset = 0;
  fftw_complex *out_moving = out;
  int n2n3 = n2 * n3;
  for (int proc = 0; proc < nprocesses; ++proc) {
    for (int i1 = 0; i1 < n1; ++i1) {
      fftw_complex *data_slice_i1 = data + i1 * n2n3;
      for (int i2 = current_n2_offset;
           i2 < current_n2_offset + partition_sizes[proc]; ++i2) {
        fftw_complex *data_slice_i1i2 = data_slice_i1 + i2 * n3;
        for (int i3 = 0; i3 < n3; ++i3) {
          *(out_moving++) = data_slice_i1i2[i3];
        }
      }
    }
    current_n2_offset += partition_sizes[proc];
  }
}

void unpartition_axis_2(fftw_complex *data, fftw_complex *out, int n1, int n2,
                        int n3, int nprocesses, int *partition_sizes) {
  int current_n2_offset = 0;
  fftw_complex *data_moving = data;
  int n2n3 = n2 * n3;
  for (int proc = 0; proc < nprocesses; ++proc) {
    for (int i1 = 0; i1 < n1; ++i1) {
      fftw_complex *out_slice_i1 = out + i1 * n2n3;
      for (int i2 = current_n2_offset;
           i2 < current_n2_offset + partition_sizes[proc]; ++i2) {
        fftw_complex *out_slice_i1i2 = out_slice_i1 + i2 * n3;
        for (int i3 = 0; i3 < n3; ++i3) {
          out_slice_i1i2[i3] = *(data_moving++);
        }
      }
    }
    current_n2_offset += partition_sizes[proc];
  }
}

void unpartition_axis_2_real(fftw_complex *data, double *out, int n1, int n2,
                             int n3, int nprocesses, int *partition_sizes,
                             double factor) {
  int current_n2_offset = 0;
  fftw_complex *data_moving = data;
  int n2n3 = n2 * n3;
  for (int proc = 0; proc < nprocesses; ++proc) {
    for (int i1 = 0; i1 < n1; ++i1) {
      double *out_slice_i1 = out + i1 * n2n3;
      for (int i2 = current_n2_offset;
           i2 < current_n2_offset + partition_sizes[proc]; ++i2) {
        double *out_slice_i1i2 = out_slice_i1 + i2 * n3;
        for (int i3 = 0; i3 < n3; ++i3) {
          out_slice_i1i2[i3] = creal(*(data_moving++)) * factor;
        }
      }
    }
    current_n2_offset += partition_sizes[proc];
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

  partition_axis_2(fft_3d_info->fft_2d_out, fft_3d_info->fft_2d_in,
                   fft_3d_info->loc_n1, fft_3d_info->n2, fft_3d_info->n3,
                   fft_3d_info->nProcesses, fft_3d_info->axis2_counts);
  send_split(fft_3d_info->fft_2d_in, fft_3d_info->fft_1d_in, fft_3d_info);

  fftw_execute(fft_3d_info->fft_1d_many);

  send_split_back(fft_3d_info->fft_1d_out, fft_3d_info->fft_2d_out,
                  fft_3d_info);
  unpartition_axis_2(fft_3d_info->fft_2d_out, out, fft_3d_info->loc_n1,
                     fft_3d_info->n3, fft_3d_info->n2, fft_3d_info->nProcesses,
                     fft_3d_info->axis2_counts);
}

void ifft_3d_2(fftw_complex *data, double *out, struct Fft3dInfo *fft_3d_info) {
  int loc_grid_size = fft_3d_info->loc_n1 * fft_3d_info->n2 * fft_3d_info->n3;
  memcpy(fft_3d_info->fft_2d_in, data, loc_grid_size * sizeof(fftw_complex));
  fftw_execute(fft_3d_info->ifft_2d_many);

  partition_axis_2(fft_3d_info->fft_2d_out, fft_3d_info->fft_2d_in,
                   fft_3d_info->loc_n1, fft_3d_info->n2, fft_3d_info->n3,
                   fft_3d_info->nProcesses, fft_3d_info->axis2_counts);
  send_split(fft_3d_info->fft_2d_in, fft_3d_info->fft_1d_in, fft_3d_info);

  fftw_execute(fft_3d_info->ifft_1d_many);

  send_split_back(fft_3d_info->fft_1d_out, fft_3d_info->fft_2d_out,
                  fft_3d_info);

  double fac = 1.0 / (fft_3d_info->n1 * fft_3d_info->n2 * fft_3d_info->n3);
  unpartition_axis_2_real(
      fft_3d_info->fft_2d_out, out, fft_3d_info->loc_n1, fft_3d_info->n2,
      fft_3d_info->n3, fft_3d_info->nProcesses, fft_3d_info->axis2_counts, fac);
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