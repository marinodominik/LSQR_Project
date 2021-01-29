#pragma once
#include "matrix.h"
#include <cublas_v2.h>

CPUMatrix cublasLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs);
CPUMatrix cublasLSQR_aux(const GPUMatrix &A, const GPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs);
void cuBLASCheck(cublasStatus_t status, int line);
void printVector(int iteration,GPUMatrix x, const char* name);