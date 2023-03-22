#include "identityMatrix.hpp"

/**
 * Initialize an identity matrix and print it in STDOUT.
 */
int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int myRows;
  double *A = initIdentityMatrix(myRank, nProcesses, myRows);

  if (myRank == 0) {
    printMatrix(A, myRows);

    int rest = N % nProcesses;
    for (int proc = 1; proc < nProcesses; ++proc) {
      if (proc == rest) {
        myRows -= 1;
      }

      MPI_Recv(A, myRows * N, MPI_DOUBLE, proc, proc, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printMatrix(A, myRows);
    }
  } else {
    MPI_Send(A, myRows * N, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  }

  delete[] A;

  MPI_Finalize();
}