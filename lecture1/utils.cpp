#include "utils.hpp"

void printMatrix(double *A, int nLoc) {
  for (int i = 0; i < nLoc; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << A[j + i * N] << " ";
    }
    std::cout << std::endl;
  }
}

void printDistributedMatrix(int myRows, double* C) {
  int myRank, nProcesses;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  if (myRank == 0) {
    printMatrix(C, myRows);

    int rest = N % nProcesses;
    for (int proc = 1; proc < nProcesses; ++proc) {
      if (proc == rest) {
        myRows -= 1;
      }

      MPI_Recv(C, myRows * N, MPI_DOUBLE, proc, proc, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printMatrix(C, myRows);
    }
  } else {
    MPI_Send(C, myRows * N, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  }
}

double *scalarAddMul(double add, double mul, double *A, int nRows) {
  double *result = new double[nRows * N];

  for (int i = 0; i < nRows; ++i) {
    for (int idx = i * N; idx < (i + 1) * N; ++idx) {
      result[idx] = mul * A[idx] + add;
    }
  }

  return result;
}