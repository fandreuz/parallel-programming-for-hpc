#pragma once

#include <mpi.h>

#include <iostream>
#define N 10
#define OUTPUT 1

void printMatrix(double *, int);
void printDistributedMatrix(int, double *);

double *scalarAddMul(double, double, double *, int);