#include "lsqrCUDAcuSparseKernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"


#define BLOCK_SIZE 32            //max threads in a block

<<<<<<< HEAD
__global__ void sqaure_vector(double *vector, const int size);
__global__ void norm2(const double *in_data, double *result, int size);
__global__ void add_subtract_vector(double *a, const double *b, const bool operation, const int size);
__global__ void scalar_vector(double *in_data, const double scalar, const int size);
__global__ void matrix_vector_multiplication(const int n_rows, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result);
=======
__global__ void sqaure_vector(const double *vector, double *result, const int size);
__global__ void norm2(const double *in_data, double *result, int elementSize);
__global__ void add_subtract_vector (double *a, const double *b, const bool operation, const int size);
__global__ void scalar_vector( double *vector, const double scalar, const int size);
__global__ void matrix_vector_multiplication(const GPUMatrix &A_sparse, const GPUMatrix &vector_dense, GPUMatrix result);
>>>>>>> fc542f3ebceac1d1af0a442a2220ea566270d687



inline unsigned int div_up(unsigned int numerator, unsigned int denominator) { //numerator = zähler, denumerator = nenner
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}



/*
<<<<<<<<<<-------------------- NORM ----------------------------->>>>>>>>>>>>>>
*/

__global__ void sqaure_vector(double *vector, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= size) { 
        return;
    } else {
        vector[i] = vector[i] * vector[i];
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
    int grids = div_up(denseVector.height, BLOCK_SIZE * BLOCK_SIZE);
    
    double *result;
    cudaMalloc(&result, grids * sizeof(double));
    
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    
    sqaure_vector<<<grids, dimBlock>>>(denseVector.elements, denseVector.height * denseVector.width); 
    norm2<<<grids, dimBlock, sh_memory_size>>>(denseVector.elements, result, denseVector.height);
    
    double *values = new double[grids]; 
    cudaMemcpy(values, result, grids * sizeof(double), cudaMemcpyDeviceToHost);
    
    double norm = 0.0;
    for (int i= 0; i< grids; i++) {
        norm += values[i];
    }

    cudaFree(result);
    delete[] values;
    return sqrt(norm);
}


/*
<<<<<<<<<<-------------------- END NORM ----------------------------->>>>>>>>>>>>>>>>>
*/

<<<<<<< HEAD
=======
__global__ void norm2(const double *in_data, double *result,int elementSize) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < elementSize){
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
>>>>>>> fc542f3ebceac1d1af0a442a2220ea566270d687




<<<<<<< HEAD
/*
<<<<<<<<<<-------------------- ADDITION AND SUBSTRACTION ----------------------------->>>>>>>>>>>>>>>>>
*/

void get_add_subtract_vector(GPUMatrix denseA, const GPUMatrix denseB, const bool operation) {
    int grids = div_up(denseA.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);

=======
void get_add_subtract_vector(GPUMatrix denseA, const GPUMatrix denseB, bool operation) {
    int grids = div_up(denseA.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
>>>>>>> fc542f3ebceac1d1af0a442a2220ea566270d687
    add_subtract_vector<<<grids, dimBlock>>>(denseA.elements, denseB.elements, operation, denseA.width * denseA.height);
}



__global__ void add_subtract_vector(double *a, const double *b, const bool operation, const int size) {
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

<<<<<<< HEAD
=======
void multiply_scalar_vector(GPUMatrix vector, const double scalar) {
>>>>>>> fc542f3ebceac1d1af0a442a2220ea566270d687



/*
<<<<<<<<<<-------------------- MULTIPLY SCALAR ----------------------------->>>>>>>>>>>>>>>>>
*/

void multiply_scalar_vector(GPUMatrix vector, const double scalar) {
    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
<<<<<<< HEAD

    scalar_vector<<<grids, dimBlock>>>(vector.elements, scalar, vector.height * vector.width);
}


__global__ void scalar_vector(double *in_data, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        in_data[i] = scalar * in_data[i];
=======
    scalar_vector<<<grids, dimBlock>>>(vector.elements, scalar, vector.height * vector.width);
    
}


__global__ void scalar_vector(double *vector, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        vector[i] = scalar * vector[i];
>>>>>>> fc542f3ebceac1d1af0a442a2220ea566270d687
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


GPUMatrix get_csr_matrix_vector_multiplication(const GPUMatrix matrix, const GPUMatrix vector) {
    GPUMatrix result = matrix_alloc_gpu(vector.height, vector.width);

    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    printf("grids: %d\n", grids);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    //int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    matrix_vector_multiplication<<<grids, dimBlock>>>(matrix.height, matrix.elements, matrix.csrRow, matrix.csrCol, vector.elements, result.elements);

    return result;
}


/*
<<<<<<<<<<-------------------- END MATRIX VECTOR MULTIPLICATION ----------------------------->>>>>>>>>>>>>>>>>
*/




GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const double lambda, const double ebs) {
    GPUMatrix x = matrix_alloc_gpu(b.height, b.width);
    double beta = getNorm2(b);



/*
    beta = norm(b);
u = b/beta;
v = A'*u;
alpha = norm(v);
v = v/alpha;
w = v;
x = 0;
phi_hat = beta;
rho_hat = alpha;
% (2) iterate
it_max = 10;
epsilon = 10^-3;
history = zeros(length(b),0);
history(:,end+1) = x;
for i = 1:it_max
    % (3) bidiagonalization
    u = A * v - alpha * u;
    beta = norm(u);
    u = u / beta;
    v = A' * u - beta * v;
    alpha = norm(v);
    v = v / alpha;
    % (4) orthogonal transformation
    rho = sqrt(rho_hat^2 + beta^2);
    c = rho_hat / rho;
    s = beta / rho;
    theta = s * alpha;
    rho_hat = -c * alpha;
    phi = c * phi_hat;
    phi_hat = s * phi_hat;
    % (5) update x, w
    x = x + (phi / rho) * w;
    w = v - (theta / rho) * w;
    history(:,end+1) = x;
    residual = norm(A*x - b);
    if(residual < epsilon)
        disp(['terminated after ',num2str(i),' iterations'])
        disp(['final residual: ',num2str(residual)])
        return
    end
end
*/



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