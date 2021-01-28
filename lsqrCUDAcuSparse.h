#pragma once
#include "matrix.h"
#include <cublas_v2.h>
#include "lsqrCUDAcuSparseKernel.h"
#include "lsqr.h"
#include "matrix.h"
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs);
CPUMatrix sparseLSQR_aux(const CPUMatrix &A, const CPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs);
void cuSPARSECheck(cusparseStatus_t status, int line);
cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A, cusparseSpMatDescr_t &A_T);
void cuSPARSECheck(cusparseStatus_t status, int line);
double sparseVectorNorm(cusparseSpVecDescr_t vector,GPUMatrix tempVector);
double normalVectorNorm(GPUMatrix src);
void scaleSparseVector(cusparseSpVecDescr_t vector,GPUMatrix tempVector,double alpha);
void scaleNormalvector(GPUMatrix src,double alpha);
void vectorAddSub(GPUMatrix a, GPUMatrix b, bool sign);
void copyFromSpVecToGpuVec(cusparseSpVecDescr_t src, GPUMatrix dst);

