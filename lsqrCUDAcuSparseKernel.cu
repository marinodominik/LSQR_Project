#include "lsqrCUDAcuSparseKernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"


#define BLOCK_SIZE 32            //max threads in a block


__global__ void norm2(const double *in_data, const int size);
__global__ void add_subtract_vector(const double *a, const double *b, double *c, const bool operation, const int size);
__global__ void scalar_vector(const double *in_data, double *out_data, const double scalar, const int size);
__global__ void add_subtract_elements_sparse_vector();
__global__ void matrix_vector_operation();



inline unsigned int div_up(unsigned int numerator, unsigned int denominator) //numerator = zähler, denumerator = nenner
{
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}



double getNorm2(const GPUMatrix denseVector) {
    return 0.0;
}



GPUMatrix get_add_subtract_vector(const GPUMatrix denseA, const GPUMatrix denseB, bool operation) {
    GPUMatrix result = matrix_alloc_gpu(denseA.height, denseA.width);

    int grids = div_up(denseA.height, BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE); //1024 threads
    add_subtract_vector<<<grids, dimBlock>>>(denseA.elements, denseB.elements, result.elements, operation, denseA.width * denseB.height);

    return result;
}



GPUMatrix multiply_scalar_vector(const GPUMatrix vector, const double scalar) {
    GPUMatrix result = matrix_alloc_gpu(vector.height, vector.width);

    int grids = div_up(vector.width, BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    scalar_vector<<<grids, dimBlock>>>(vector.elements, result.elements, scalar, vector.height * vector.width);
    
    return result;
}



// <<<<<<<<<<< Vector ist in dense format >>>>>>>>>>>>>>>>>>>
__global__ void norm2(const double *in_data, const int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * 2 + blockDim.x;
    sdata[tid] = in_data[i];
    __syncthreads();

    sdata[tid] = (in_data[i] * in_data[i]) + (in_data[i + blockDim.x] * in_data[i + blockDim.x]);
    __syncthreads();
}



__global__ void add_subtract_vector(const double *a, const double *b, double *c, const bool operation, const int size) {
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



__global__ void scalar_vector(const double *in_data, double *out_data, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        out_data[i] = scalar * in_data[i];
    }
    __syncthreads();
}


//shared memory
__global__ void matrix_vector_operation(const GPUMatrix &A_sparse, const GPUMatrix &vector_dense, GPUMatrix result) {

}


GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const double lambda, const double ebs) {

    return u;

}



CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const double lambda, const double ebs) {
    CPUMatrix resultCPU = matrix_alloc_cpu(b.height, b.width);
    GPUMatrix resultGPU = matrix_alloc_gpu(b.height, b.width);

    //GPUMatrix gpuA = matrix_alloc_sparse_gpu(A.height, A.width, A.elementSize, A.rowSize, A.columnSize);
    GPUMatrix A_gpu = matrix_alloc_gpu(b.height, b.width);
    GPUMatrix b_gpu = matrix_alloc_gpu(b.height, b.width);
    
    /* upload Matrix, vector */
    //matrix_upload_cuSparse(A, gpuA);
    matrix_upload(b, A_gpu);
    matrix_upload(b, b_gpu);

    resultGPU = lsqr_algrithm(A_gpu, b_gpu, lambda, ebs);

    printVector(b.height * b.width, resultGPU, "u");

    /* Download result */
    matrix_download(resultGPU, resultCPU);

    /* free GPU memory */
    cudaFree(resultGPU.elements);
    cudaFree(A_gpu.elements);
    cudaFree(b_gpu.elements);

    return resultCPU;
}