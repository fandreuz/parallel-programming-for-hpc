#include "identityMatrix.hpp"
#include <string>

#if MODE == 2
#include <cblas.h>
#endif

/**
 * A is mem-copied immediately to the device, B is moved as soon as we get a
 * columns block. We allocate immediately space on the device for the full C,
 * and we mem-copy C to the host after the whole computation.
 */
#if MODE == 3
#include <cublas_v2.h>
#endif

/**
 * Distributed matrix multiplication.
 */
int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

#if MODE == 2
  openblas_set_num_threads(1);
#endif

  int myRows;
  double *A = initIdentityMatrix(myRank, nProcesses, myRows);
  int myRowsB;
  double *B = initIdentityMatrix(myRank, nProcesses, myRowsB);
  if (myRows != myRowsB) {
    std::cerr << "An error occurred" << std::endl;
    return 1;
  }

  int remainder = SIZE % nProcesses;
  int div = SIZE / nProcesses;

  int *splits = new int[nProcesses];
  for (int proc = 0; proc < nProcesses; ++proc) {
    splits[proc] = div + (proc < remainder);
  }
  int *shifted_cumsum_splits = new int[nProcesses];
  shifted_cumsum_splits[0] = 0;
  for (int proc = 0; proc < nProcesses; ++proc) {
    shifted_cumsum_splits[proc] =
        shifted_cumsum_splits[proc - 1] + splits[proc - 1];
  }

  double *A2 = scalarAddMul(1, 2, A, myRows);
  double *B2 = scalarAddMul(5, 2, B, myRows);
  delete[] A;
  delete[] B;

  double *C = new double[myRows * SIZE];
#if !defined(CUDACC) &&                                                        \
    MODE != 2 // no need to initialize to zero with DGEMM or CUDA
  memset(C, 0, myRows * SIZE * sizeof(double));
#endif

#if MODE == 3 // load A in the accelerator and initialize the cublas context
  auto cublas_handle = cublasCreate();
  double *dev_a;
  int a_memory_size = myRows * SIZE * sizeof(double);
  cudaMalloc((void **)&dev_a, a_memory_size);
  cudaMemcpy(dev_a, a, a_memory_size, cudaMemcpyHostToDevice);

  double *dev_b;
  cudaMalloc((void **)&dev_b, SIZE * splits[0] * sizeof(double));

  double *dev_c;
  cudaMalloc((void **)&dev_c, myRows * SIZE * sizeof(double));
#endif

  // 0: communication preparation
  // 1: communication
  // 2: computation
  std::vector<double> times(3, 0.0);
  double checkpoint1, checkpoint2, checkpoint3;

  // splits[0] is the maximum number of columns of B we will ever send
  double *B_send_buffer = new double[myRows * splits[0]];
  double *B_col_block = new double[SIZE * splits[0]];
  double *B_row0 = B2;

  for (int proc = 0; proc < nProcesses; ++proc) {
    int n_cols_B_sent = splits[proc];
    int *recv_count = new int[nProcesses];
    for (int p = 0; p < nProcesses; ++p) {
      recv_count[p] = n_cols_B_sent * splits[p];
    }
    int *displ = new int[nProcesses];
    displ[0] = 0;
    for (int p = 1; p < nProcesses; ++p) {
      displ[p] = displ[p - 1] + recv_count[p - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    checkpoint1 = MPI_Wtime();

#if MODE == 0
    // Neither row nor col-major order, values in the same column within a
    // single process, and are scattered in small segments when Allgatherv is
    // called.
    for (int B_loc_col = 0; B_loc_col < n_cols_B_sent; ++B_loc_col) {
      double *B_send_buffer_col = B_send_buffer + B_loc_col * myRows;
      for (int B_loc_row = 0; B_loc_row < myRows; ++B_loc_row) {
        B_send_buffer_col[B_loc_row] = B_row0[B_loc_row * SIZE];
      }
      // move along B's row 0
      ++B_row0;
    }
#else
    // row-major order
    double *B_ptr = B_row0;
    double *B_send_buffer_write = B_send_buffer;
    for (int B_loc_row = 0; B_loc_row < myRows; ++B_loc_row) {
      for (int B_loc_col = 0; B_loc_col < n_cols_B_sent; ++B_loc_col) {
        *B_send_buffer_write++ = *B_ptr++;
      }
      B_ptr = B_row0 + (B_loc_row + 1) * SIZE;
    }
    B_row0 += n_cols_B_sent;
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    checkpoint2 = MPI_Wtime();
    MPI_Allgatherv(B_send_buffer, myRows * n_cols_B_sent, MPI_DOUBLE,
                   B_col_block, recv_count, displ, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    checkpoint3 = MPI_Wtime();

// find top-left corner of the block of C we're writing
#if MODE == 3
    double *C_write = dev_c + shifted_cumsum_splits[proc];
    cudaMemcpy(dev_b, B_col_block, SIZE * n_cols_B_sent * sizeof(double),
               cudaMemcpyHostToDevice);
#else
    double *C_write = C + shifted_cumsum_splits[proc];
#endif

#if MODE == 3
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(CblasRowMajor, CUBLAS_OP_T, CUBLAS_OP_T, myRows, n_cols_B_sent,
                SIZE, &alpha, dev_a, SIZE, dev_b, n_cols_B_sent, &beta, C_write,
                SIZE);
#elif MODE == 0
    double *A_loc_row = A2;
    for (int A_loc_row_idx = 0; A_loc_row_idx < myRows; ++A_loc_row_idx) {
      for (int B_block_col_idx = 0; B_block_col_idx < n_cols_B_sent;
           ++B_block_col_idx) {
        for (int p = 0; p < nProcesses; ++p) {
          double *A_loc_proc_row = A_loc_row + shifted_cumsum_splits[p];
          double *B_col_block_p =
              B_col_block + B_block_col_idx * n_cols_B_sent + displ[p];
          for (int p_row = 0; p_row < n_cols_B_sent; ++p_row) {
            *C_write += A_loc_proc_row[p_row] * B_col_block_p[p_row];
          }
        }
        ++C_write;
      }
      // jump to the next line
      C_write += SIZE - n_cols_B_sent;
      A_loc_row += SIZE;
    }
#elif MODE == 1 // row-major
    for (int A_loc_row_idx = 0; A_loc_row_idx < myRows; ++A_loc_row_idx) {
      double *A_row = A2 + A_loc_row_idx * SIZE;
      double *B_col_row0 = B_col_block;
      for (int B_block_col_idx = 0; B_block_col_idx < n_cols_B_sent;
           ++B_block_col_idx) {
        for (int k = 0, kB = 0; k < SIZE; ++k, kB += n_cols_B_sent) {
          *C_write += A_row[k] * B_col_row0[kB];
        }
        ++C_write;
        ++B_col_row0;
      }
      // jump to the next line
      C_write += SIZE - n_cols_B_sent;
    }
#elif MODE == 2
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, myRows,
                n_cols_B_sent, SIZE, 1.0, A2, SIZE, B_col_block, n_cols_B_sent,
                0.0, C_write, SIZE);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    // save times
    times[2] += MPI_Wtime() - checkpoint3;
    times[1] += checkpoint3 - checkpoint2;
    times[0] += checkpoint2 - checkpoint1;
  }

  delete[] B_send_buffer;
  delete[] B_col_block;
  delete[] splits;
  delete[] shifted_cumsum_splits;

#if MODE == 3
  cudaMemcpy(C, dev_c, myRows * SIZE * sizeof(double), cudaMemcpyDeviceToHost);
#endif

#ifdef OUTPUT
  if (myRank == 0) {
    std::cout << "A" << std::endl;
  }
  printDistributedMatrix(myRows, A2);

  if (myRank == 0) {
    std::cout << "B" << std::endl;
  }
  printDistributedMatrix(myRows, B2);

  if (myRank == 0) {
    std::cout << "C" << std::endl;
  }
  printDistributedMatrix(myRows, C);
#endif

  delete[] A2;
  delete[] B2;
  delete[] C;

  if (myRank == 0) {
    std::ofstream proc_out;
    proc_out.open("proc" + std::to_string(myRank) + ".out", std::ios_base::app);

    write_to_file(times, proc_out);

    proc_out.close();
  }

#if MODE == 3
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cublasDestroy(cublas_handle);
#endif

  MPI_Finalize();
}