#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

const double h = 0.1;

void save_gnuplot(FILE *file, double *M, size_t myRows, size_t dim);

void evolve(double *matrix, double *matrix_new, size_t myRows,
            size_t dimension);

int above_peer(int myRank);
int below_peer(int myRank, int nProcesses);
size_t compute_my_rows(int myRank, int dimension, int nProcesses);

int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  int aboveRank = above_peer(myRank);
  int belowRank = below_peer(myRank, nProcesses);

#ifdef _OPENACC
  int devtype = acc_get_device_type();
  int devNum = acc_get_num_devices(devtype);
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

  int byte_dimension = sizeof(double) * (myRows + 2) * (dimension + 2);
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

#pragma acc data copy(matrix) copyin(matrix_new)
  {
#pragma acc parallel
    {
      for (size_t it = 0; it < iterations; ++it) {
        evolve(matrix, matrix_new, myRows, dimension);

        double *tmp_matrix = matrix;
        matrix = matrix_new;
        matrix_new = tmp_matrix;

#pragma acc host_data use_device(matrix)
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
  }

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

size_t compute_my_rows(int myRank, int dimension, int nProcesses) {
  size_t myRows = dimension / nProcesses;
  myRows += myRank < dimension % nProcesses;
  return myRows;
}

int above_peer(int myRank) {
  if (myRank == 0) {
    return MPI_PROC_NULL;
  }
  return myRank - 1;
}

int below_peer(int myRank, int nProcesses) {
  if (myRank == nProcesses - 1) {
    return MPI_PROC_NULL;
  }
  return myRank + 1;
}

void evolve(double *matrix, double *matrix_new, size_t myRows,
            size_t dimension) {
#pragma acc loop
  for (size_t i = 1; i <= myRows; ++i)
    for (size_t j = 1; j <= dimension; ++j)
      matrix_new[(i * (dimension + 2)) + j] =
          (0.25) * (matrix[((i - 1) * (dimension + 2)) + j] +
                    matrix[(i * (dimension + 2)) + (j + 1)] +
                    matrix[((i + 1) * (dimension + 2)) + j] +
                    matrix[(i * (dimension + 2)) + (j - 1)]);
}

void save_gnuplot(FILE *file, double *M, size_t myRows, size_t dimension) {
  for (size_t i = 1; i < myRows + 1; ++i)
    for (size_t j = 0; j < dimension + 2; ++j)
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i,
              M[(i * (dimension + 2)) + j]);
}
