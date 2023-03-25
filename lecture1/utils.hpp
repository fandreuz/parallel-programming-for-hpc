#pragma once

#include <mpi.h>

#include <iostream>
#define N 10000
#define OUTPUT 0

void printMatrix(double *, int);
void printDistributedMatrix(int, double *);

double *scalarAddMul(double, double, double *, int);