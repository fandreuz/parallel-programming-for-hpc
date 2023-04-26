#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

const double h = 0.1;

void evolve(double *matrix_new, double *matrix, size_t myRows,
            size_t dimension);

int computeAbovePeer(int myRank);
int computeBelowPeer(int myRank, int nProcesses);
size_t compute_my_rows(int myRank, int dimension, int nProcesses);

void save_gnuplot(FILE *file, double *M, size_t myRows, size_t dim);

int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int aboveRank = computeAbovePeer(myRank);
  int belowRank = computeBelowPeer(myRank, nProcesses);

#ifdef _OPENACC
  int devtype = acc_get_device_type();
  int devNum = acc_get_num_devices(devtype);
  int dev = myRank % devNum;
  acc_set_device_num(dev, devtype);
#endif

  size_t dimension = 0, iterations = 0;

  if (argc != 3) {
    if (myRank == 0) {
      fprintf(stderr, "\nwrong number of arguments. Usage: ./a.out dim it\n");
    }

    MPI_Finalize();
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);

  if (myRank == 0) {
    printf("matrix size = %zu\n", dimension);
    printf("number of iterations = %zu\n", iterations);
  }

  size_t myRows = compute_my_rows(myRank, dimension, nProcesses);

  double *matrix, *matrix_new;

  // borders
  double increment = 100.0 / (dimension + 1);
  double incrementStart = increment * dimension * (double)myRank / nProcesses;
  if (myRank > dimension % nProcesses) {
    incrementStart += (dimension % nProcesses) * increment;
  } else {
    incrementStart += myRank * increment;
  }

  int matrixElementsCount = (myRows + 2) * (dimension + 2);
  int byte_dimension = sizeof(double) * matrixElementsCount;
  matrix = (double *)malloc(byte_dimension);
  matrix_new = (double *)malloc(byte_dimension);

  memset(matrix, 0, byte_dimension);
  memset(matrix_new, 0, byte_dimension);

  for (size_t i = 1; i <= myRows; ++i)
    for (size_t j = 1; j <= dimension; ++j)
      matrix[(i * (dimension + 2)) + j] = 0.5;

  for (size_t i = 1; i <= myRows + 1; ++i) {
    matrix[i * (dimension + 2)] = i * increment + incrementStart;
    matrix_new[i * (dimension + 2)] = i * increment + incrementStart;
  }

  for (size_t i = 1; i <= dimension + 1; ++i) {
    matrix[((myRows + 1) * (dimension + 2)) + (dimension + 1 - i)] =
        i * increment;
    matrix_new[((myRows + 1) * (dimension + 2)) + (dimension + 1 - i)] =
        i * increment;
  }

  int rowSize = 1 + dimension + 1;

  int recvTopIdx = 1;
  int sendTopIdx = recvTopIdx + rowSize;

  int sendBottomIdx = myRows * rowSize + 1;
  int recvBottomIdx = sendBottomIdx + rowSize;

  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

#pragma acc data copy(matrix [0:matrixElementsCount])                          \
    copyin(matrix_new [0:matrixElementsCount])
  {
    for (size_t it = 0; it < iterations / 2; ++it) {

      evolve(matrix_new, matrix, myRows, dimension);

#pragma acc host_data use_device(matrix, matrix_new)
      {
        MPI_Sendrecv(matrix_new + sendTopIdx, dimension, MPI_DOUBLE, aboveRank,
                     0, matrix_new + recvBottomIdx, dimension, MPI_DOUBLE,
                     belowRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(matrix_new + sendBottomIdx, dimension, MPI_DOUBLE,
                     belowRank, 0, matrix_new + recvTopIdx, dimension,
                     MPI_DOUBLE, aboveRank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
      }

      evolve(matrix, matrix_new, myRows, dimension);

#pragma acc host_data use_device(matrix, matrix_new)
      {
        MPI_Sendrecv(matrix + sendTopIdx, dimension, MPI_DOUBLE, aboveRank, 0,
                     matrix + recvBottomIdx, dimension, MPI_DOUBLE, belowRank,
                     0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(matrix + sendBottomIdx, dimension, MPI_DOUBLE, belowRank,
                     0, matrix + recvTopIdx, dimension, MPI_DOUBLE, aboveRank,
                     0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_end = MPI_Wtime();

  if (myRank == 0)
    printf("\nelapsed time = %f seconds\n", t_end - t_start);

  if (myRank == 0) {
    FILE *file = fopen("solution.dat", "w");

    // top
    for (size_t j = 0; j < dimension + 2; ++j)
      fprintf(file, "%f\t%f\t%f\n", h * j, 0, matrix[j]);

    save_gnuplot(file, matrix, myRows, dimension);

    int procRows = myRows;
    for (int proc = 1; proc < nProcesses; ++proc) {
      procRows = compute_my_rows(proc, dimension, nProcesses);
      MPI_Recv(matrix, (procRows + 2) * (dimension + 2), MPI_DOUBLE, proc, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      save_gnuplot(file, matrix, procRows, dimension);
    }

    // bottom
    for (size_t j = 0; j < dimension + 2; ++j)
      fprintf(file, "%f\t%f\t%f\n", h * j, 0,
              matrix[(procRows + 1) * (dimension + 2) + j]);
    fclose(file);
  } else {
    MPI_Send(matrix, (myRows + 2) * (dimension + 2), MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}

void evolve(double *matrix_new, double *matrix, size_t myRows,
            size_t dimension) {
#pragma acc parallel loop
  for (size_t i = 1; i <= myRows; ++i)
    for (size_t j = 1; j <= dimension; ++j)
      matrix[(i * (dimension + 2)) + j] =
          (0.25) * (matrix_new[((i - 1) * (dimension + 2)) + j] +
                    matrix_new[(i * (dimension + 2)) + (j + 1)] +
                    matrix_new[((i + 1) * (dimension + 2)) + j] +
                    matrix_new[(i * (dimension + 2)) + (j - 1)]);
}

size_t compute_my_rows(int myRank, int dimension, int nProcesses) {
  size_t myRows = dimension / nProcesses;
  myRows += myRank < dimension % nProcesses;
  return myRows;
}

int computeAbovePeer(int myRank) {
  if (myRank == 0) {
    return MPI_PROC_NULL;
  }
  return myRank - 1;
}

int computeBelowPeer(int myRank, int nProcesses) {
  if (myRank == nProcesses - 1) {
    return MPI_PROC_NULL;
  }
  return myRank + 1;
}

void save_gnuplot(FILE *file, double *M, size_t myRows, size_t dimension) {
  for (size_t i = 1; i < myRows + 1; ++i)
    for (size_t j = 0; j < dimension + 2; ++j)
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i,
              M[(i * (dimension + 2)) + j]);
}
