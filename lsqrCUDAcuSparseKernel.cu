#include "lsqrCUDAcuSparseKernel.h"
#include "lsqr.h"
#include "matrix.h"
#include <cusparse.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.h"

#DEFINE BLOCK_SIZE 32           //max threads in a block


double getNorm2(const GPUMatrix denseVector) {
    
}

GPUMatrix get_add_subtract_vector(const GPUMatrix denseA, const GPUMatrix denseB) {

}

GPUMatrix multiply_scalar_vector(const GPUMatrix vector, const double scalar) {

}



// <<<<<<<<<<< Vector ist in dense format >>>>>>>>>>>>>>>>>>>
__global__ norm2(const double *in_data, const int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * 2  blockDim.x;
    sdata[tid] = in_data[i];
    __syncthreads();

    sdata[tid] = (in_data[i] * in_data[i]) + (in_data[i + blockDim.x] * in_data[i + blockDim.x]);
    __syncthreads();
}

__global__ add_subtract_elements_vector(const double *a, const double *b, const double *c, const bool operation, const int size) {
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


__global__ scalar_vector(const double *in_data, const double *out_data, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        out_data[i] = scalar * in_data[i];
    }
    __syncthreads();
}


// Kernel for matrix Sparse Format
__global__ add_subtract_elements_sparse_vector() {

}

//shared memory
__global__ matrix_vector_operation() {

}




CPUMatrix sparseLSQR_with_kernels(const GPUMatrix &A, const GPUMatrix &b, double lambda, double ebs) {

    GPUMatrix result = matrix_alloc_gpu(b.height, b.width);

    int scalar_grids = div_up(m.width, BLOCK_SIZE);
    dim3 dimBlock(1024);
    scalar_vector<<<scalar_grids, dimBlock>>>(b.elements, result.elements, 0.5, b.width * b.height);
    for (int i = 0; i < b.width * b.height; i ++) std::cout << result.elements[i] << ", ";
}