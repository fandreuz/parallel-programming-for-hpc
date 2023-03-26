#pragma once

#include <mpi.h>

#include <iostream>

void printMatrix(double *, int);
void printDistributedMatrix(int, double *);

double *scalarAddMul(double, double, double *, int);