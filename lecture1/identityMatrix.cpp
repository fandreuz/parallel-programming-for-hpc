#include "identityMatrix.hpp"

double *initIdentityMatrix(int myRank, int nProcesses, int &myRows) {
  myRows = N / nProcesses;
  int remainder = N % nProcesses;

  int offset = 0;
  if (myRank < remainder) {
    myRows++;
  } else {
    offset = remainder;
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