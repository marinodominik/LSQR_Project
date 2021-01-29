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
#include <stdio.h>


CPUMatrix cusparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs);
CPUMatrix cusparseLSQR_aux(const CPUMatrix &A, const GPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs);

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A);


void cuSPARSECheck(cusparseStatus_t status, int line);
double normalVectorNorm(cusparseDnVecDescr_t src, GPUMatrix temp);
void scaleNormalvector(cusparseDnVecDescr_t src,double alpha,GPUMatrix temp);
void vectorAddSub(cusparseDnVecDescr_t a, cusparseDnVecDescr_t b, bool sign,GPUMatrix temp);
void copyVector(cusparseDnVecDescr_t dst,cusparseDnVecDescr_t src,GPUMatrix temp);
void printDenseVector(cusparseDnVecDescr_t src,const char* name,GPUMatrix temp);
void printNormalVector(GPUMatrix x, const char* name);