#include "identityMatrix.hpp"
#include <fstream>
#include <string>
#include <vector>

void write_to_file(const std::vector<double> &, std::ofstream &);

/**
 * Distributed matrix multiplication (assuming N % nProcesses == 0).
 */
int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  if (N % nProcesses != 0) {
    std::cerr << "Remainder is not zero" << std::endl;
    return 1;
  }

  int myRows;
  double *A = initIdentityMatrix(myRank, nProcesses, myRows);
  int myRowsB;
  double *B = initIdentityMatrix(myRank, nProcesses, myRowsB);
  if (myRows != myRowsB) {
    std::cerr << "An error occurred" << std::endl;
    return 1;
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

  int small_square = myRows * myRows;
  double *B_send_buffer = new double[small_square];
  double *B_col_block = new double[myRows * N];
  double *B_row0 = B2;
  for (int proc = 0; proc < nProcesses; ++proc) {
    checkpoint1 = MPI_Wtime();
    for (int B_loc_col = 0; B_loc_col < myRows; ++B_loc_col) {
      double *B_send_buffer_col = B_send_buffer + B_loc_col * myRows;
      for (int B_loc_row = 0; B_loc_row < myRows; ++B_loc_row) {
        // values in the same column are adjacent
        B_send_buffer_col[B_loc_row] = B_row0[B_loc_row * N];
      }
      // move along B's row 0
      ++B_row0;
    }

    checkpoint2 = MPI_Wtime();
    MPI_Allgather(B_send_buffer, small_square, MPI_DOUBLE, B_col_block,
                  small_square, MPI_DOUBLE, MPI_COMM_WORLD);

    checkpoint3 = MPI_Wtime();
    double *C_write = C + myRows * proc;
    double *A_loc_row = A2;
    for (int A_loc_row_idx = 0; A_loc_row_idx < myRows; ++A_loc_row_idx) {
      double *B_block_row0 = B_col_block;
      for (int B_block_col_idx = 0; B_block_col_idx < myRows;
           ++B_block_col_idx) {
        for (int p = 0; p < nProcesses; ++p) {
          double *A_loc_proc_row = A_loc_row + p * myRows;
          double *B_block_proc_row0 = B_block_row0 + p * small_square;
          for (int p_row = 0; p_row < myRows; ++p_row) {
            *C_write += A_loc_proc_row[p_row] * B_block_proc_row0[p_row];
          }
        }
        ++C_write;
        B_block_row0 += myRows;
      }
      C_write += N - myRows;
      A_loc_row += N;
    }

    // save times
    comp_times.push_back(MPI_Wtime() - checkpoint3);
    comm_times.push_back(checkpoint3 - checkpoint2);
    comm_setup_times.push_back(checkpoint2 - checkpoint1);
  }

  delete[] B_send_buffer;
  delete[] B_col_block;

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
  for (int i = 0; i < vec.size(); ++i) {
    file << vec[i] << " ";
  }
  file << std::endl;
}