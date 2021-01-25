#pragma once
#include <iostream>
#include <tuple>
#include <fstream>
#include "matrix.h"

/* <<<<<--------- READ DATA IN COMPRESED SPARSE ROW FORMAT ----------->>>>>>>>>> */
CPUMatrix read_matrix_in_csr(const char *path);
CPUMatrix read_data_in_csr(const char *path);

double *swapping_d_vector(double *elements, int elementSize);
int *swapping_i_vector(int *elements, int elementSize);
double *end_d_vector(double *elements, int size);
int *end_i_vector(int *elements, int size);