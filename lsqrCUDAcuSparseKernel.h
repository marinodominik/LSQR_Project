#pragma once
#pragma once
#include "matrix.h"
#include <cublas_v2.h>


CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, double lambda, double ebs);