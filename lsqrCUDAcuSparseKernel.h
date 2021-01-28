#pragma once

#include "matrix.h"

double getNorm2(const GPUMatrix &denseVector);

GPUMatrix get_add_subtract_vector(const GPUMatrix denseA, const GPUMatrix denseB, bool operation);
GPUMatrix multiply_scalar_vector(const GPUMatrix vector, const double scalar);

CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const double lambda, const double ebs);

GPUMatrix lsqr_algrithm();

inline unsigned int div_up(unsigned int numerator, unsigned int denominator);