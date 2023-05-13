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

  printDistributedMatrix(myRows, A);

  delete[] A;

  MPI_Finalize();
}