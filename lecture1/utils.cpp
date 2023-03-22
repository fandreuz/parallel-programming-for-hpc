#include "utils.hpp"

void printMatrix(double *A, int nLoc) {
  for (int i = 0; i < nLoc; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << A[j + i * N] << " ";
    }
    std::cout << std::endl;
  }
}