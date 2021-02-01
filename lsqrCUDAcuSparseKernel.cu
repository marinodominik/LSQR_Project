#include "lsqrCUDAcuSparseKernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"


#define BLOCK_SIZE 32            //max threads in a block

__global__ void sqaure_vector(const double *vector, double *result, const int size);
__global__ void norm2(const double *in_data, double *result);
__global__ void add_subtract_vector(const double *a, const double *b, double *c, const bool operation, const int size);
__global__ void scalar_vector(const double *in_data, double *out_data, const double scalar, const int size);
__global__ void matrix_vector_multiplication(const GPUMatrix &A_sparse, const GPUMatrix &vector_dense, GPUMatrix result);



inline unsigned int div_up(unsigned int numerator, unsigned int denominator) { //numerator = zähler, denumerator = nenner
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}



__global__ void sqaure_vector(const double *vector, double *result, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= size) { 
        return;
    } else {
        result[i] = vector[i] * vector[i];
    }

    __syncthreads();
}



double getNorm2(const GPUMatrix denseVector) {
    GPUMatrix tmp = matrix_alloc_gpu(denseVector.height, denseVector.width);

    int grids = div_up(denseVector.height, BLOCK_SIZE * BLOCK_SIZE);
    
    double *result = new double[grids];
    cudaMalloc(&result, grids * sizeof(double));
    
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    
    sqaure_vector<<<grids, dimBlock>>>(denseVector.elements, tmp.elements, tmp.height * tmp.width); 
    norm2<<<grids, dimBlock, sh_memory_size>>>(tmp.elements, result);
    
    double *values = new double[grids]; 
    cudaMemcpy(values, result, grids * sizeof(double), cudaMemcpyDeviceToHost);
    
    double norm = 0.0;
    for (int i= 0; i< grids; i++) {
        norm += values[i];
    }

    return sqrt(norm);
}



__global__ void norm2(const double *in_data, double *result) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    sdata[tid] = in_data[i];        //load global data in sh_memory
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    //thread 0 writes in result back to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0]; //Da wir n-grids haben, werden die zahlen für jeden block in eine eigene zelle im global gespeichert
    }
}




GPUMatrix get_add_subtract_vector(const GPUMatrix denseA, const GPUMatrix denseB, bool operation) {
    GPUMatrix result = matrix_alloc_gpu(denseA.height, denseA.width);

    int grids = div_up(denseA.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    add_subtract_vector<<<grids, dimBlock>>>(denseA.elements, denseB.elements, result.elements, operation, denseA.width * denseA.height);

    return result;
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



GPUMatrix multiply_scalar_vector(const GPUMatrix vector, const double scalar) {
    GPUMatrix result = matrix_alloc_gpu(vector.height, vector.width);

    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    scalar_vector<<<grids, dimBlock>>>(vector.elements, result.elements, scalar, vector.height * vector.width);
    
    return result;
}


__global__ void scalar_vector(const double *in_data, double *out_data, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        out_data[i] = scalar * in_data[i];
    }
    __syncthreads();
}




//shared memory
__global__ void matrix_vector_multiplication(const int n_rows, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows) {
        const int row_start = rowPtr[row];
        const int row_end = rowPtr[row + 1];

        double sum = 0.0;
        for (int idx = row_start; idx < row_end; idx ++) {
            int col = colIdx[idx];
            sum += elements[idx] * x[col];
        }
        printf("sum: %lf\n", sum);
        result[row] = sum;
    }
    __syncthreads();
}


GPUMatrix get_csr_matrix_vector_multiplication(const GPUMatrix matrix, const GPUMatrix vector) {
    GPUMatrix result = matrix_alloc_gpu(vector.height, vector.width);

    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    //int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    matrix_vector_multiplication<<<grids, dimBlock>>>(matrix.height, matrix.elements, matrix.csrRow, matrix.csrCol, vector.elements, result.elements);

    return result;
}




GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const double lambda, const double ebs) {
    GPUMatrix result = matrix_alloc_gpu(b.height, b.width);
    result = get_csr_matrix_vector_multiplication(A, b);
    return result; 
}



CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const double lambda, const double ebs) {
    CPUMatrix resultCPU = matrix_alloc_cpu(b.height, b.width);
    GPUMatrix resultGPU = matrix_alloc_gpu(b.height, b.width);

    GPUMatrix A_gpu = matrix_alloc_sparse_gpu(A.height, A.width, A.elementSize, A.rowSize, A.columnSize);
    GPUMatrix b_gpu = matrix_alloc_gpu(b.height, b.width);
    
    /* upload Matrix, vector */
    matrix_upload_cuSparse(A, A_gpu);
    matrix_upload(b, b_gpu);

    resultGPU = lsqr_algrithm(A_gpu, b_gpu, lambda, ebs);

    //printVector(b.height * b.width, resultGPU, "add vector");

    /* Download result */
    matrix_download(resultGPU, resultCPU);

    /* free GPU memory */
    cudaFree(resultGPU.elements);
    cudaFree(A_gpu.elements);
    cudaFree(b_gpu.elements);

    return resultCPU;
}