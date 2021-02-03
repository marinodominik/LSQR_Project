#pragma once
#include <iostream>
#include "matrix.h"
/*#include "lsqrCUDAcuBlas.h"*/
#include "math.h"


double getNorm2(const GPUMatrix denseVector);
void get_add_subtract_vector(GPUMatrix denseA, GPUMatrix denseB, bool operation);
void multiply_scalar_vector(const GPUMatrix vector, const double scalar);
GPUMatrix get_csr_matrix_vector_multiplication(const GPUMatrix A_sparse, const GPUMatrix b_dense);

void kernelCheck(int line);
void printValuesKernel(GPUMatrix x, const char *name); 

void printVectorKernel(int iteration,GPUMatrix x, const char* name);

GPUMatrix transpose_matrix(GPUMatrix A);

CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const double lambda, const double ebs);
GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const double lambda, const double ebs);

inline unsigned int div_up(unsigned int numerator, unsigned int denominator);