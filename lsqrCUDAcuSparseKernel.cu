#include "lsqrCUDAcuSparseKernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include <cusparse_v2.h>

#define BLOCK_SIZE 32            //max threads in a block

__global__ void sqaure_vector(const double *vector, double *tmp, const int size);
__global__ void norm2(const double *in_data, double *result, int size);
__global__ void add_subtract_vector(double *a, double *b, bool operation, int size);  
__global__ void scalar_vector(double *in_data, const double scalar, const int size);
__global__ void matrix_vector_multiplication(const int n_rows, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result);
__global__ void matrix_vector_multiplication_sh(const int n_row, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result);


inline unsigned int div_up(unsigned int numerator, unsigned int denominator) { //numerator = zähler, denumerator = nenner
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}


GPUMatrix transpose_matrix(GPUMatrix A) {
    GPUMatrix A_transpose = matrix_alloc_sparse_gpu(A.height, A.width, A.elementSize, A.rowSize, A.columnSize);
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    size_t tempInt;
    double *buffer;

    cusparseCsr2cscEx2_bufferSize(handle, A.height, A.width, A.elementSize,
                                  A.elements, A.csrRow, A.csrCol,
                                  A_transpose.elements, A_transpose.csrCol,A_transpose.csrRow, 
                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &tempInt);

    cudaMalloc(&buffer, tempInt);

    cusparseCsr2cscEx2(handle, A.height, A.width, A.elementSize,
                       A.elements, A.csrRow, A.csrCol,
                       A_transpose.elements, A_transpose.csrRow, A_transpose.csrCol, 
                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
    
    return A_transpose;
}



/*
<<<<<<<<<<-------------------- NORM ----------------------------->>>>>>>>>>>>>>
*/

__global__ void sqaure_vector(const double *vector, double *tmp, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= size) { 
        return;
    } else {
        tmp[i] = vector[i] * vector[i];
    }

    __syncthreads();
}


__global__ void norm2(const double *in_data, double *result,int size) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < size){
        sdata[tid] = in_data[i];        //load global data in sh_memory
    }else{
        sdata[tid] = 0; 
    }
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


double getNorm2(const GPUMatrix denseVector) {
    GPUMatrix tmp = matrix_alloc_gpu(denseVector.height, denseVector.width);

    int grids = div_up(denseVector.height, BLOCK_SIZE * BLOCK_SIZE);

    double *result;
    cudaMalloc(&result, grids * sizeof(double));
    
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    
    sqaure_vector<<<grids, dimBlock>>>(denseVector.elements, tmp.elements, tmp.height * tmp.width); 
    norm2<<<grids, dimBlock, sh_memory_size>>>(tmp.elements, result, tmp.height * tmp.width);

    
    double *values = new double[grids]; 
    cudaMemcpy(values, result, grids * sizeof(double), cudaMemcpyDeviceToHost);

    double norm = 0.0;
    for (int i= 0; i< grids; i++) {
        norm += values[i];
    }

    matrix_free_gpu(tmp);
    delete[] values;
    cudaFree(result);

    return sqrt(norm);
}


/*
<<<<<<<<<<-------------------- END NORM ----------------------------->>>>>>>>>>>>>>>>>
*/





/*
<<<<<<<<<<-------------------- ADDITION AND SUBSTRACTION ----------------------------->>>>>>>>>>>>>>>>>
*/

void get_add_subtract_vector(GPUMatrix denseA, GPUMatrix denseB, bool operation) {
    printf("get add\n");
    int grids = div_up(denseA.height, BLOCK_SIZE * BLOCK_SIZE);
    kernelCheck(__LINE__);
    printf("%d\n", grids);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    kernelCheck(__LINE__);
    printf("before");
    kernelCheck(__LINE__);
    add_subtract_vector<<<grids, dimBlock>>>(denseA.elements, denseB.elements, operation, denseA.width * denseA.height);
    kernelCheck(__LINE__);
    printf("after");
}



__global__ void add_subtract_vector(double *a, double *b, bool operation, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    //check if index out of range of vector
    if(i >= size) return;

    if(operation == true) {
        a[i] = a[i] + b[i];

    } else {
        a[i] = a[i] - b[i];
    }
    __syncthreads();
}


/*
<<<<<<<<<<-------------------- END ADDITON AND SUBSTRACTION ----------------------------->>>>>>>>>>>>>>>>>
*/




/*
<<<<<<<<<<-------------------- MULTIPLY SCALAR ----------------------------->>>>>>>>>>>>>>>>>
*/

void multiply_scalar_vector(GPUMatrix vector, const double scalar) {
    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);

    scalar_vector<<<grids, dimBlock>>>(vector.elements, scalar, vector.height * vector.width);
}


__global__ void scalar_vector(double *in_data, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        in_data[i] = scalar * in_data[i];
    }
    __syncthreads();
}
/*
<<<<<<<<<<-------------------- END MULTIPLICATION SCALAR ----------------------------->>>>>>>>>>>>>>>>>
*/




/*
<<<<<<<<<<-------------------- CSR MATRIX MULTIPLY WITH DENSE VECTOR ----------------------------->>>>>>>>>>>>>>>>>
*/

__global__ void matrix_vector_multiplication(const int n_rows, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows) {
        const int row_start = rowPtr[row];
        const int row_end = rowPtr[row + 1];

        double sum = 0.0;
        for (int idx = row_start; idx < row_end; idx++) {
            int col = colIdx[idx];
            sum += elements[idx] * x[col];
        }
        result[row] = sum;
    }
    __syncthreads();
}

__global__ void matrix_vector_multiplication_sh(const int n_row, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result) {
    
}


GPUMatrix get_csr_matrix_vector_multiplication_sh(const GPUMatrix matrix, const GPUMatrix vector) {
    GPUMatrix result = matrix_alloc_gpu(vector.height, vector.width);

    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);

    matrix_vector_multiplication_sh<<<grids, dimBlock, sh_memory_size>>>(matrix.height, matrix.elements, matrix.csrRow, matrix.csrCol, vector.elements, result.elements);

    return result;
}



GPUMatrix get_csr_matrix_vector_multiplication(const GPUMatrix matrix, const GPUMatrix vector) {
    GPUMatrix result = matrix_alloc_gpu(vector.height, vector.width);

    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    matrix_vector_multiplication<<<grids, dimBlock>>>(matrix.height, matrix.elements, matrix.csrRow, matrix.csrCol, vector.elements, result.elements);

    return result;
}


/*
<<<<<<<<<<-------------------- END MATRIX VECTOR MULTIPLICATION ----------------------------->>>>>>>>>>>>>>>>>
*/


GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const double lambda, const double ebs) {
    GPUMatrix x = matrix_alloc_gpu(b.height, b.width);
    printValuesKernel(A, "A");

    GPUMatrix A_transpose = transpose_matrix(A);

    printValuesKernel(A_transpose, "A_transpose");

    return x; 
}


CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const double lambda, const double ebs) {
    CPUMatrix resultCPU = matrix_alloc_cpu(b.height, b.width);
    GPUMatrix resultGPU = matrix_alloc_gpu(b.height, b.width);

    GPUMatrix A_gpu = matrix_alloc_sparse_gpu(A.height, A.width, A.elementSize, A.rowSize, A.columnSize);
    GPUMatrix b_gpu = matrix_alloc_gpu(b.height, b.width);

    /* upload Matrix, vector */
    matrix_upload_cuSparse(A, A_gpu);
    matrix_upload(b, b_gpu);
    
    printf("hier1\n");
    resultGPU = lsqr_algrithm(A_gpu, b_gpu, lambda, ebs);
    printf("hier2");

    /* Download result */
    matrix_download(resultGPU, resultCPU);

    /* free GPU memory */
    matrix_free_sparse_gpu(A_gpu);
    matrix_free_gpu(b_gpu);
    matrix_free_gpu(resultGPU);

    return resultCPU;
}


void printVectorKernel(int iteration,GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height, x.width);
	matrix_download(x ,tempCPUMatrix);
	//printf("iteration number: %d\n", iteration);
	for(int i = 0; i < 9; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}

void printValuesKernel(GPUMatrix x, const char *name) {
    printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_sparse_cpu(x.height, x.width, x.elementSize, x.rowSize, x.columnSize);
    matrix_download_cuSparse(x ,tempCPUMatrix);
    

    for(int i = 0; i < 9; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
    }
    printf("\n Row:");

    for(int i = 0; i < 4; i++){
		printf("%d ", tempCPUMatrix.csrRow[i]);
    }
    printf("\n Col:");
    for(int i = 0; i < 9; i++){
		printf("%d ", tempCPUMatrix.csrCol[i]);
    }
    printf("\n");
}


void kernelCheck(int line){
	const cudaError_t err = cudaGetLastError();                            
	if (err != cudaSuccess) {                                              
    	const char *const err_str = cudaGetErrorString(err);               
    	std::cerr << "Cuda error in " << __FILE__ << ":" << line - 1   
            << ": " << err_str << " (" << err << ")" << std::endl;   
            exit(EXIT_FAILURE);                                                                    
	}
}