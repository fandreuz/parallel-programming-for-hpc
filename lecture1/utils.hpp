#pragma once

#include <mpi.h>

#include <iostream>
#include <vector>
#include <fstream>

void printMatrix(double *, int);
void printDistributedMatrix(int, double *);

double *scalarAddMul(double, double, double *, int);

void write_to_file(const std::vector<double> &, std::ofstream &);