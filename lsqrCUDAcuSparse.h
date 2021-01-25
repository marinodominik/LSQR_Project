#pragma once
#include "matrix.h"
#include <cublas_v2.h>


CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, double lambda, double ebs);

CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs);
CPUMatrix sparseLSQR_aux(const cusparseSpMatDescr_t &A, const cusparseSpVecDescr_t &b,cusparseDnVecDescr_t &u,cusparseDnVecDescr_t &v,cusparseDnVecDescr_t &w,cusparseDnVecDescr_t &x,cusparseDnVecDescr_t &tempVector,double ebs);
void cuSPARSECheck(cusparseStatus_t status, int line);