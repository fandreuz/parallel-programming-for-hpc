#include "utils.hpp"

void printMatrix(double *A, int nLoc) {
  for (int i = 0; i < nLoc; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << A[j + i * N] << " ";
    }
    std::cout << std::endl;
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