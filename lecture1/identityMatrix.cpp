#include "identityMatrix.hpp"
#include "utils.hpp"
#include <mpi.h>
#include <string>

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

double *initIdentityMatrix(int myRank, int nProcesses, int &myRows) {
  myRows = N / nProcesses;
  int rest = N % nProcesses;

  int offset = 0;
  if (myRank < rest) {
    myRows++;
  } else {
    offset = rest;
  }

  double *A = new double[N * myRows];
  memset(A, 0, N * myRows * sizeof(double));

  int firstRow = myRows * myRank + offset;
  for (int i = 0; i < myRows; ++i) {
    // column in the global matrix
    int col = firstRow + i;
    // row offset in the local (linearized) matrix
    int rowOffset = i * N;
    A[col + rowOffset] = 1.0;
  }

  return A;
}