#include "lsqrCUDAcuSparse.h"

CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs){
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__line);
    GPUMatrix gpuVectorb = matrix_alloc_gpu(b.height,b.width);
    matrix_upload(b,gpuVectorb);
    GPUMatrix u = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix v = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix w = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix x = matrix_alloc_gpu(b.height,b.width);
	GPUMatrix tempVector = matrix_alloc_gpu(b.height,b.width);
	CPUMatrix fillingToX = matrix_alloc_cpu(b.height,b.width);
    for(int i=0;i<b.height;i++){
		fillingToX.elements[i]=0;
	}
	cuSPARSECheck(status,__LINE__); 
	matrix_upload(fillingToX,x);  

    cusparseDestroy(handle);
    return sparseLSQR_aux()  
}

CPUMatrix sparseLSQR_aux(const GPUMatrix &A, const GPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__line)

}

void cuSPARSECheck(cusparseStatus_t status, int line){
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}