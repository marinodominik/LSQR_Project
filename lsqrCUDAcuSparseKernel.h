#pragma once

#include "matrix.h"
#include "lsqrCUDAcuBlas.h"
#include "math.h"

double getNorm2(const GPUMatrix denseVector);
void get_add_subtract_vector(const GPUMatrix denseA, const GPUMatrix denseB, bool operation);
void multiply_scalar_vector(const GPUMatrix vector, const double scalar);
GPUMatrix get_csr_matrix_vector_multiplication(const GPUMatrix A_sparse, const GPUMatrix b_dense);

CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const double lambda, const double ebs, const int max_iters);
GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const double lambda, const double ebs, const int max_iters);

inline unsigned int div_up(unsigned int numerator, unsigned int denominator);