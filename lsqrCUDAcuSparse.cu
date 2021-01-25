#include "lsqrCUDAcuSparse.h"
#include "lsqr.h"
#include "matrix.h"
#include <cusparse.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs){
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__line);
    cusparseSpVecDescr_t spVectorb;
    GPUMatrix u = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix v = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix w = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix x = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix tempVector = matrix_alloc_gpu(b.height,b.width); 
    status = cusparseCreateSpVec(&spVectorb,b.height,b.elementSize,b.csrCol,b.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    CPUMatrix res = sparseLSQR_aux(spMatrixA,spVectorb,u,v,w,x,tempVector,A.height,A.width,A.elementSize,ebs);
    cusparseClean(handle,spMatrixA,spVectorb,u,v,w,x,tempVector);
    return res; 
}

CPUMatrix sparseLSQR_aux(const CPUMatrix &A, const cusparseSpVecDescr_t &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
    int tempInt;
    double *buffer;
    GPUMatrix tempGpuVec = matrix_alloc_gpu(b.height,1);
    cusparseSpMatDescr_t spMatrixA,spMatrixA_T;
    status = cusparseCreateCsr(&A,A.height,A.rows,A.elementSize,A.csrRow,A.csrCol,A.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    //create A_T
    cusparseStatus_t status;
    cusparseHandle_t handle;
    spStatus = cusparseCreate(&spHandle);
    cuSPARSECheck(spStatus,__LINE__);
    prev_err = 100000000; 
	//init stage
    //beta = norm(b)
    beta = sparseVectorNorm(b,tempGpuVec);
    //u = b/beta
    copyFromSpVecToVec(b,u);
    scaleNormalvector(u,beta);
    //v = A'*u
    tempDouble = 1; tempDouble2 = 0;
    status = cusparseCsrmvEx_bufferSize(spHandle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A_T.height,A_T.rows,A_T.elementSize,&tempDouble,CUDA_R_64F,A,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,u.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&tempInt);
    cudaMalloc(&buffer, tempInt);
    status = cusparseCsrmvEx(spHandle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A_T.height,A_T.rows,A_T.elementSize,&tempDouble,CUDA_R_64F,A,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,u.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&buffer);
    //alpha = norm(v)
    alpha = normalVectorNorm(v);
    //v = v/alpha;
    scaleNormalvector(v,alpha);
    //w = v;
    cudaMemcpy (w.elements, v.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
	phi_tag = beta; rho_tag = alpha;
	int i = 0, counter = 0;
	while(true){
		//next bidiagonlization
        // u = A * v - alpha * u;
        tempDouble = 1; tempDouble2 = (-1)*alpha;
        status = cusparseCsrmvEx(spHandle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A.height,A.rows,A.elementSize,&tempDouble,CUDA_R_64F,A,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,v.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,u.elements,A_T.height,CUDA_R_64F,&buffer);
        //beta = norm(u);
        beta = normalVectorNorm(u);
        // u = u / beta;
        scaleNormalvector(u,beta);
        // v = A' * u - beta * v;
        tempDouble = 1; tempDouble2 = (-1)*beta;
        status = cusparseCsrmvEx(spHandle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A_T.height,A_T.rows,A_T.elementSize,&tempDouble,CUDA_R_64F,A,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,u.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&buffer);
        //alpha = norm(v)
        alpha = normalVectorNorm(v);
        //v = v/alpha;
        scaleNormalvector(v,alpha);
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
        cudaMemcpy (tempVector.elements, w.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
        scaleNormalvector(tempVector,phi/rho); 
        //x = (phi / rho) * w + x;             (in cublas : x is y, w is x)
        vectorAddSub(x, tempVector,true)
        //	w = -(theta / rho) * w + v;
        cudaMemcpy (tempVector.elements, w.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
        scaleNormalvector(w,(theta/rho)); 
        vectorAddSub(w,v,false)
        //check for convergence
        tempDouble = 1; tempDouble2 = (0);
        status = cusparseCsrmvEx(spHandle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A.height,A.rows,A.elementSize,&tempDouble,CUDA_R_64F,A,A.elements,CUDA_R_64F,A.csrRow,A.csrCol,x.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&buffer);
		//residual = norm(A*x - b);
        //Ax - b (result in tempVector)
    }

}

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A,  cusparseSpVecDescr_t &b,cusparseDnVecDescr_t &u,cusparseDnVecDescr_t &v,cusparseDnVecDescr_t &w,cusparseDnVecDescr_t &x,cusparseDnVecDescr_t &tempVector){
    status = cusparseDestroySpMat(A);
    status = cusparseDestroySpVec(b)
    status = cusparseDestroy(handle);
    cuSPARSECheck(status,__LINE__);
}


void cuSPARSECheck(cusparseStatus_t status, int line){
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

double sparseVectorNorm(cusparseSpVecDescr_t vector,GPUMatrix temp){
    double res = 0.0; 
    cusparseSpVecGetValues(vector,tempVector.elements);
    //kernel call on temp vector, result in res;
    return 0.0;
}
double normalVectorNorm(GPUMatrix src){
    double res = 0.0; 
    //kernel call on src, result in res;
    return 0.0;
}
void scaleSparseVector(cusparseSpVecDescr_t vector,GPUMatrix temp,double alpha){
    cusparseSpVecGetValues(vector,tempVector.elements);
    //kernel call on temp vector, new vector in tempVector
    cusparseSpVecSetValues(vector,tempVector);
}
void scaleNormalvector(GPUMatrix src,double alpha){
    //kernel call on src vector
}
void vectorAddSub(GPUMatrix a, GPUMatrix b, boolean sign){
    GPUMatrix tempGpuVec = matrix_alloc_gpu(a.height,1);
    // some kernel call, result in tempgpu vecv
    cudaMemcpy (a.elements, tempGpuVec.elements, a.height*sizeof(double), cudaMemcpyDeviceToDevice);
}
void copyFromSpVecToGpuVec(cusparseSpVecDescr_t src, GPUMatrix dst){
    cusparseStatus_t status = cusparseSpVecGetValues(vector,dst.elements);
    cuSPARSECheck(status,__LINE__);
}



/*
    cusparseAxpby() - for vector addition
    cusparseCsrmvEx() matrix-vector multiplication
    cublas norm

    A is an m×n sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); x and y are vectors;

*/