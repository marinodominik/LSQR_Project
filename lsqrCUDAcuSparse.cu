#include "lsqrCUDAcuSparse.h"
#include "lsqr.h"
#include "matrix.h"
#include <cusparse.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


// <<<<<<<<<<< Vector ist in dense format >>>>>>>>>>>>>>>>>>>
__global__ square_elements(double *in_data, double *out_data) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = out_data[i];
    __syncthreads();

    for ()

    __syncthreads();
}


__global__ norm(double *in_data, double *out_data) {

}


__global__ add_subtract_elements_vector(double *in_Adata, double *in_Bdata, bool operation, double *out_data) {
    //
}


__global__ scalar_vector() {


}


// <<<<<<<<<<<<<<<<<<<<<< END 

// Kernel for matrix Sparse Format

__global__ add_subtract_elements_sparse_vector() {


}


__global__ matrix_vector_operation() {
    std::cout <<"hi";
}




CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, double lambda, double ebs) {


}


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


/*
    cusparseAxpby() - for vector addition
    


*/