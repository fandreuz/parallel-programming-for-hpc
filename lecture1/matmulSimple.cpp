#include "identityMatrix.hpp"

/**
 * Distributed matrix multiplication (rest not taken into account).
 */
int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  if (N % nProcesses != 0) {
    std::cerr << "Rest is not zero" << std::endl;
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
  double *B2 = scalarAddMul(0, 2, B, myRows);
  delete[] A, B;

  double *C = new double[myRows * N];
  memset(C, 0, N * myRows * sizeof(double));

  int small_square = myRows * myRows;
  double *B_send_buffer = new double[small_square];
  double *B_col_block = new double[myRows * N];
  for (int proc = 0; proc < nProcesses; ++proc) {
    for (int j = 0; j < myRows; ++j) {
      for (int row = 0; row < myRows; ++row) {
        // values in the same column are adjacent
        B_send_buffer[row + j * myRows] = B2[j + proc * myRows + row * N];
      }
    }

    MPI_Allgather(B_send_buffer, small_square, MPI_DOUBLE, B_col_block,
                  small_square, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < myRows; ++i) {
      for (int j = 0; j < myRows; ++j) {
        int c_idx = j + proc * myRows + i * N;
        for (int k = 0; k < N; ++k) {
          C[c_idx] += A2[k + i * N] * B_col_block[(k % myRows) + j * myRows +
                                                  (k / myRows) * small_square];
        }
      }
    }
  }

  delete[] B_send_buffer, B_col_block;

  delete[] A2, B2;

  if (myRank == 0) {
    printMatrix(C, myRows);

    int rest = N % nProcesses;
    for (int proc = 1; proc < nProcesses; ++proc) {
      if (proc == rest) {
        myRows -= 1;
      }

      MPI_Recv(C, myRows * N, MPI_DOUBLE, proc, proc, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printMatrix(A, myRows);
    }
  } else {
    MPI_Send(A, myRows * N, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  }

  delete[] C;

  MPI_Finalize();
}