#pragma once
#include "matrix.h"
#include <cublas_v2.h>

CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs);
CPUMatrix sparseLSQR_aux(const GPUMatrix &A, const GPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs);
void cuSPARSECheck(cusparseStatus_t status, int line);