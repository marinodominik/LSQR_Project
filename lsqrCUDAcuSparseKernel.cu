#include "lsqrCUDAcuSparseKernel.h"
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