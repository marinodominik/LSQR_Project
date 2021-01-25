#include <stdio.h>
#include "lsqr.h"
#include "lsqrCUDAcuBlas.h"
#include "matrix.h"

#include <cusparse.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> //cuda basic linear algebra subroutine library
#define BLOCK_SIZE 32

__global__ void sparseLSQRKernel();//decleration 
__global__ void normalLSQRKernel();//decleration
/*
	A normal LSQR implemtation - GPU Matrix A, Vector b (as 1*n matrix)
	u,v,w,x are matrix that is allocated before hand to use for cuBLAS computations 
	vector x needs to be initialzed to 0.
*/
CPUMatrix normalLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs){
	cublasHandle_t handle;
	cublasStatus_t status;
	status = cublasCreate(&handle);

    GPUMatrix tempGpuMatrixA = matrix_alloc_gpu(A.height,A.width);
	matrix_upload(A,tempGpuMatrixA);
	GPUMatrix gpuMatrixA = matrix_alloc_gpu(A.height,A.width);
	matrix_upload(A,gpuMatrixA);

	//TRANSPOSE A
	double tempDouble = 1.0;
	double tempDouble2 = 0.0;
	cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,A.height,A.width,&tempDouble,tempGpuMatrixA.elements,A.width,&tempDouble2,tempGpuMatrixA.elements,A.width,gpuMatrixA.elements,A.width);
	cuBLASCheck(status,__LINE__); 

	matrix_free_gpu(tempGpuMatrixA);

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
	cuBLASCheck(status,__LINE__); 
	matrix_upload(fillingToX,x);
	cublasDestroy(handle);
	cuBLASCheck(status,__LINE__); 

	return normalLSQR_aux(gpuMatrixA,gpuVectorb,u,v,w,x,tempVector,ebs);
}

CPUMatrix normalLSQR_aux(const GPUMatrix &A, const GPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs){
	double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
	cublasHandle_t handle;
	cublasStatus_t status;
	cuBLASCheck(status,__LINE__); 
	status = cublasCreate(&handle);
	prev_err = 100000000; 
	cuBLASCheck(status,__LINE__);
	//init stage
	//beta = norm(b)
	status = cublasDnrm2(handle, b.height, b.elements,1,&beta); 
	cuBLASCheck(status,__LINE__);
	//u = b/beta
	cudaMemcpy (u.elements, b.elements, b.height*sizeof(double), cudaMemcpyDeviceToDevice);
	tempDouble = 1/beta;
	status = cublasDscal(handle, u.height,&tempDouble,u.elements,1);
	cuBLASCheck(status,__LINE__);
	//printVector(-1,x,"X");
	//printVector(-1,u,"U");
	//v = A'*u
	tempDouble = 0.0;
	tempDouble2 = 1.0;
	status = cublasDgemv (handle, CUBLAS_OP_T, A.width, A.height,&tempDouble2,A.elements, A.width, u.elements,1,&tempDouble, v.elements, 1);
	cuBLASCheck(status,__LINE__);
	//printVector(-1,v,"V");
	//alpha = norm(v)
	status = cublasDnrm2(handle, v.height, v.elements,1,&alpha); 
	cuBLASCheck(status,__LINE__);
	//v = v/alpha;
	tempDouble = 1/alpha;
	status = cublasDscal(handle, v.height,&tempDouble,v.elements,1);
	cuBLASCheck(status,__LINE__);
	printVector(-1,v,"V");
	//w = v;
	cudaMemcpy (w.elements, v.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
	phi_tag = beta; rho_tag = alpha;
	
	//printf("constants: alpha: %.6f beta:%.6f\n",alpha,beta);

	int i = 0, counter = 0;
	while(true){
		//next bidiagonlization
		// u = A * v - alpha * u;
		tempDouble = alpha*(-1.0);
		tempDouble2 = 1.0;
		status = cublasDgemv (handle, CUBLAS_OP_N, A.width,A.height,&tempDouble2,A.elements, A.width, v.elements,1,&tempDouble, u.elements, 1);
		cuBLASCheck(status,__LINE__);
		//beta = norm(u);
		status = cublasDnrm2(handle, u.height, u.elements,1,&beta); 
		cuBLASCheck(status,__LINE__);
		// u = u / beta;
		tempDouble = 1/beta;
		status = cublasDscal(handle, u.height,&tempDouble,u.elements,1);
		cuBLASCheck(status,__LINE__);
		// v = A' * u - beta * v;
		tempDouble = (-1.0)*beta;
		tempDouble2 = 1.0;
		status = cublasDgemv (handle, CUBLAS_OP_T, A.width,A.height,&tempDouble2,A.elements, A.width, u.elements,1, &tempDouble, v.elements, 1);
		cuBLASCheck(status,__LINE__);
		//alpha = norm(v)
		status = cublasDnrm2(handle, v.height, v.elements,1,&alpha); 
		cuBLASCheck(status,__LINE__);
		//v = v/alpha;
		tempDouble = 1/alpha;
		status = cublasDscal(handle, v.height,&tempDouble,v.elements,1);
		cuBLASCheck(status,__LINE__);
		//next orthogonal transformation
		rho = sqrt(pow (rho_tag, 2.0) + pow (beta, 2.0));
		c = rho_tag / rho;
		s = beta / rho;
		theta = s * alpha;
		rho_tag = (-1) * c * alpha;
		phi = c * phi_tag;
		phi_tag = s * phi_tag;
		//printf("constants: alpha: %.6f beta:%.6f\n",alpha,beta);
		//printf("constants: rho: %.6f c: %.6f s: %.6f theta: %.6f rho_tag: %.6f phi: %.6f\n phi_tag: %.6f\n",rho,c,s,theta,rho_tag,phi,phi_tag);
		//updating x,w
		//x =  (phi / rho) * w + x;             (in cublas : x is y, w is x)
		tempDouble = phi / rho;
		status = cublasDaxpy(handle,w.height,&tempDouble,w.elements, 1,x.elements, 1);
		cuBLASCheck(status,__LINE__);
		//printVector(i, x,"X");
		//	w = v - (theta / rho) * w ;
		tempDouble = (-1.0) * (theta / rho);
		cudaMemcpy (tempVector.elements, v.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
		status = cublasDaxpy(handle,tempVector.height,&tempDouble,w.elements, 1,tempVector.elements, 1);
		cuBLASCheck(status,__LINE__);
		cudaMemcpy (w.elements, tempVector.elements, tempVector.height*sizeof(double), cudaMemcpyDeviceToDevice);
		//printVector(i, w,"W");
		//check for convergence
		//residual = norm(A*x - b);
		cudaMemcpy (tempVector.elements, b.elements, tempVector.height*sizeof(double), cudaMemcpyDeviceToDevice);
		//Ax - b (result in tempVector)
		tempDouble = -1.0;
		tempDouble2 = 1.0;
		status = cublasDgemv (handle, CUBLAS_OP_N, A.width,A.height,&tempDouble2,A.elements, A.width, x.elements,1,&tempDouble, tempVector.elements, 1);
		cuBLASCheck(status,__LINE__);
		status = cublasDnrm2(handle, tempVector.height,tempVector.elements,1,&curr_err); 
		cuBLASCheck(status,__LINE__);
		improvment = prev_err-curr_err;
		printf("line: %d size of error: %.6f improvment of: %.6f\n",i,curr_err,improvment);i++;
		if(improvment<ebs) counter++; else counter = 0;
		if(counter>1000) break;
		prev_err = curr_err;
	}
	CPUMatrix result = matrix_alloc_cpu(x.height,x.width);
	matrix_download(x,result);
	cublasDestroy(handle);
	return result;
}

__global__ void normalLSQRKernel(){}


__global__ void sparseLSQRKernel(){/*
	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int tx = blockx*BLOCK_SIZE +threadx;
	int ty = blocky*BLOCK_SIZE +thready;*/
    
}
void cuBLASCheck(cublasStatus_t status, int line){
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

/*
general operation of the LSQR Code:
Based on the matlab implemntation of the algorithm,
that is based on the give article in pages 50/51. 

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
disp(['it_max (=',num2str(it_max),') reached'])
disp(['final residual: ',num2str(residual)])
end

*/



/*
GENERAL CUSPARSE MATRIX OPERATIONS:

cusparse<t>[<matrix data format>]<operation>[<output matrix data format>]

WHERE:
-> <t> can be S, D, C, Z, or X, corresponding to the data types float, double, cuComplex,
cuDoubleComplex, and the generic type, respectively.

-> <matrix data format> can be dense, coo, csr, or csc, corresponding to the dense,
coordinate, compressed sparse row, and compressed sparse column formats, respectively.

-> <operation> can be axpyi, gthr, gthrz, roti, or sctr, corresponding to the
Level 1 functions; it also can be mv or sv, corresponding to the Level 2 functions, as well as mm
or sm, corresponding to the Level 3 functions.

->axpyi:
->gthr:
->gthrz:
->roti:
->sctr:
*/

void printVector(int iteration,GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height,x.width);
	matrix_download(x,tempCPUMatrix);
	printf("iteration number: %d\n", iteration);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}