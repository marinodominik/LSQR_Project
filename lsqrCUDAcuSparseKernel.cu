#include "lsqrCUDAcuSparseKernel.h"
#include "lsqr.h"
#include "matrix.h"
#include <cusparse.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.h"

#DEFINE BLOCK_SIZE 32           //max threads in a block


// <<<<<<<<<<< Vector ist in dense format >>>>>>>>>>>>>>>>>>>
__global__ norm2(double *in_data, double *out_data, int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * 2  blockDim.x;
    sdata[tid] = in_data[i];
    __syncthreads();

    sdata[tid] = in_data[i] * in_data[i] + in_data[i + blockDim.x] * in_data[i + blockDim.x];
    __syncthreads();
}

__global__ add_subtract_elements_vector(double *a, double *b, double *c, bool operation, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    //check if index out of range of vector
    if(i >= size) return;

    if(operation == true) {
        c[i] = a[i] + b[i];

    } else {
        c[i] = a[i] - b[i];
    }
    __syncthreads();
}


__global__ scalar_vector(double *in_data, double *out_data, double scalar, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        out_data[i] = scalar * in_data[i];
    }
    __syncthreads();
}

// Kernel for matrix Sparse Format
__global__ add_subtract_elements_sparse_vector() {


}


__global__ matrix_vector_operation() {
    std::cout <<"hi";
}




CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, double lambda, double ebs) {
    int scalar_grids = div_up(m.width, BLOCK_SIZE);
    dim3 dimBlock(1024);
    scalar_vector<<<scalar_grids, dimBlock>>>();       //in ein block können nur 1024 (32 * 32 ) Threads
}