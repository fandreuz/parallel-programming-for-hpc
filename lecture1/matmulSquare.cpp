#include "identityMatrix.hpp"
#include <fstream>
#include <string>
#include <vector>

void write_to_file(const std::vector<double> &, std::ofstream &);

/**
 * Distributed matrix multiplication.
 */
int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int myRows;
  double *A = initIdentityMatrix(myRank, nProcesses, myRows);
  int myRowsB;
  double *B = initIdentityMatrix(myRank, nProcesses, myRowsB);
  if (myRows != myRowsB) {
    std::cerr << "An error occurred" << std::endl;
    return 1;
  }

  int remainder = N % nProcesses;
  int div = N / nProcesses;

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

  double *C = new double[myRows * N];
  memset(C, 0, N * myRows * sizeof(double));

  std::vector<double> comm_setup_times;
  std::vector<double> comm_times;
  std::vector<double> comp_times;
  double checkpoint1, checkpoint2, checkpoint3;

  // splits[0] is the maximum number of columns of B we will ever send
  double *B_send_buffer = new double[myRows * splits[0]];
  double *B_col_block = new double[N * splits[0]];
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

    checkpoint1 = MPI_Wtime();
    for (int B_loc_col = 0; B_loc_col < n_cols_B_sent; ++B_loc_col) {
      double *B_send_buffer_col = B_send_buffer + B_loc_col * myRows;
      for (int B_loc_row = 0; B_loc_row < myRows; ++B_loc_row) {
        // values in the same column are adjacent
        B_send_buffer_col[B_loc_row] = B_row0[B_loc_row * N];
      }
      // move along B's row 0
      ++B_row0;
    }

    checkpoint2 = MPI_Wtime();
    MPI_Allgatherv(B_send_buffer, myRows * n_cols_B_sent, MPI_DOUBLE,
                   B_col_block, recv_count, displ, MPI_DOUBLE, MPI_COMM_WORLD);

    checkpoint3 = MPI_Wtime();
    double *C_write = C + shifted_cumsum_splits[proc];
    double *A_loc_row = A2;
    for (int A_loc_row_idx = 0; A_loc_row_idx < myRows; ++A_loc_row_idx) {
      for (int B_block_col_idx = 0; B_block_col_idx < n_cols_B_sent;
           ++B_block_col_idx) {
        for (int p = 0; p < nProcesses; ++p) {
          double *A_loc_proc_row = A_loc_row + shifted_cumsum_splits[p];
          double *B_col_block_p =
              B_col_block + B_block_col_idx * splits[p] + displ[p];
          for (int p_row = 0; p_row < splits[p]; ++p_row) {
            *C_write += A_loc_proc_row[p_row] * B_col_block_p[p_row];
          }
        }
        if (myRank == 0) {
          std::cout << std::endl << std::endl;
        }
        ++C_write;
      }
      C_write += N - n_cols_B_sent;
      A_loc_row += N;
    }

    // save times
    comp_times.push_back(MPI_Wtime() - checkpoint3);
    comm_times.push_back(checkpoint3 - checkpoint2);
    comm_setup_times.push_back(checkpoint2 - checkpoint1);
  }

  delete[] B_send_buffer;
  delete[] B_col_block;
  delete[] splits;
  delete[] shifted_cumsum_splits;

  if (OUTPUT) {
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
  }
  delete[] A2;
  delete[] B2;
  delete[] C;

  std::ofstream proc_out;
  proc_out.open("proc" + std::to_string(myRank) + ".out");

  write_to_file(comm_setup_times, proc_out);
  write_to_file(comm_times, proc_out);
  write_to_file(comp_times, proc_out);

  proc_out.close();

  MPI_Finalize();
}

void write_to_file(const std::vector<double> &vec, std::ofstream &file) {
  for (std::size_t i = 0; i < vec.size(); ++i) {
    file << vec[i] << " ";
  }
  file << std::endl;
}