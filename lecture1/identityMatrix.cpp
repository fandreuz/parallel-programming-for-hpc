#include <iostream>
#include <mpi.h>
#include <string>
#define N 10

void printMatrix(double *, int);

/**
 * Initialize an identity matrix and print it in STDOUT.
 */
int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int myRows = N / nProcesses;
  int rest = N % nProcesses;

  int offset = 0;
  if (myRank < rest) {
    myRows++;
  } else {
    offset = rest;
  }

  double *A = (double *)malloc(N * myRows * sizeof(double));
  memset(A, 0, N * myRows * sizeof(double));

  int firstRow = myRows * myRank + offset;
  for (int i = 0; i < myRows; ++i) {
    // column in the global matrix
    int col = firstRow + i;
    // row offset in the local (linearized) matrix
    int rowOffset = i * N;
    A[col + rowOffset] = 1.0;
  }

  if (myRank == 0) {
    printMatrix(A, myRows);
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

  MPI_Finalize();
}

void printMatrix(double *A, int nLoc) {
  for (int i = 0; i < nLoc; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << A[j + i * N] << " ";
    }
    std::cout << std::endl;
  }
}