#pragma once
#include "matrix.h"
#include <cublas_v2.h>


CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs);
CPUMatrix sparseLSQR_aux(const CPUMatrix &A, const cusparseSpVecDescr_t &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs);

void cuSPARSECheck(cusparseStatus_t status, int line);

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A,  cusparseSpVecDescr_t &b,cusparseDnVecDescr_t &u,cusparseDnVecDescr_t &v,cusparseDnVecDescr_t &w,cusparseDnVecDescr_t &x,cusparseDnVecDescr_t &tempVector){
    status = cusparseDestroySpMat(A);
    status = cusparseDestroySpVec(b)
    status = cusparseDestroy(handle);
    cuSPARSECheck(status,__LINE__);
}


void cuSPARSECheck(cusparseStatus_t status, int line){
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

double sparseVectorNorm(cusparseSpVecDescr_t vector,GPUMatrix temp);
double normalVectorNorm(GPUMatrix src);
void scaleSparseVector(cusparseSpVecDescr_t vector,GPUMatrix temp,double alpha);
void scaleNormalvector(GPUMatrix src,double alpha);
void vectorAddSub(GPUMatrix a, GPUMatrix b, boolean sign);
void copyFromSpVecToGpuVec(cusparseSpVecDescr_t src, GPUMatrix dst);

