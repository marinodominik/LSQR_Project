#pragma once
#pragma once
#include "matrix.h"
#include <cublas_v2.h>

double getNorm2(const GPUMatrix denseVector);

GPUMatrix get_add_subtract_vector(const GPUMatrix denseA, const GPUMatrix denseB);
GPUMatrix multiply_scalar_vector(const GPUMatrix vector, const double scalar);

CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, double lambda, double ebs);