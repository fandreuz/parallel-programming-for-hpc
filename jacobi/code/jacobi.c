#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

void save_gnuplot(double *M, size_t dim);

void evolve(double *matrix, double *matrix_new, size_t dimension);

int main(int argc, char *argv[]) {
  int myRank, nProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;

  if (argc != 3) {
    if (myRank == 0) {
      fprintf(stderr,
              "\nwrong number of arguments. Usage: ./a.out dim it\n");
    }

    MPI_Finalize();
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);

  if (myRank == 0) {
    printf("matrix size = %zu\n", dimension * nProcesses);
    printf("number of iterations = %zu\n", iterations);
  }

  double *matrix, *matrix_new;

  byte_dimension = sizeof(double) * (dimension + 2) * (dimension + 2);
  matrix = (double *)malloc(byte_dimension);
  matrix_new = (double *)malloc(byte_dimension);

  memset(matrix, 0, byte_dimension);
  memset(matrix_new, 0, byte_dimension);

  // initial values
  for (size_t i = 1; i <= dimension; ++i)
    for (size_t j = 1; j <= dimension; ++j)
      matrix[(i * (dimension + 2)) + j] = 0.5;

  // borders
  double increment = 100.0 / (dimension + 1);

  for (size_t i = 1; i <= dimension + 1; ++i) {
    matrix[i * (dimension + 2)] = i * increment;
    matrix[((dimension + 1) * (dimension + 2)) + (dimension + 1 - i)] =
        i * increment;
    matrix_new[i * (dimension + 2)] = i * increment;
    matrix_new[((dimension + 1) * (dimension + 2)) + (dimension + 1 - i)] =
        i * increment;
  }

  double t_start = MPI_Wtime();
  for (size_t it = 0; it < iterations; ++it) {
    evolve(matrix, matrix_new, dimension);

    double *tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  }
  double t_end = MPI_Wtime();

  printf("\nelapsed time = %f seconds\n", t_end - t_start);

  save_gnuplot(matrix, dimension);

  free(matrix);
  free(matrix_new);

  MPI_Finalize();

  return 0;
}

void evolve(double *matrix, double *matrix_new, size_t dimension) {
  for (size_t i = 1; i <= dimension; ++i)
    for (size_t j = 1; j <= dimension; ++j)
      matrix_new[(i * (dimension + 2)) + j] =
          (0.25) * (matrix[((i - 1) * (dimension + 2)) + j] +
                    matrix[(i * (dimension + 2)) + (j + 1)] +
                    matrix[((i + 1) * (dimension + 2)) + j] +
                    matrix[(i * (dimension + 2)) + (j - 1)]);
}

const double h = 0.1;
void save_gnuplot(double *M, size_t dimension) {
  FILE *file;

  file = fopen("solution.dat", "w");

  for (size_t i = 0; i < dimension + 2; ++i)
    for (size_t j = 0; j < dimension + 2; ++j)
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i,
              M[(i * (dimension + 2)) + j]);

  fclose(file);
}
