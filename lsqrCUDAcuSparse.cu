#include "lsqrCUDAcuSparse.h"


CPUMatrix sparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs){
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__LINE__);
    GPUMatrix u = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix v = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix w = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix x = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix tempVector = matrix_alloc_gpu(b.height,b.width); 
    CPUMatrix res = sparseLSQR_aux(A,b,u,v,w,x,tempVector,ebs);
    return res; 
}

CPUMatrix sparseLSQR_aux(const CPUMatrix &A, const CPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
    size_t tempInt;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__LINE__);
    prev_err = 100000000; 
    double *buffer;
    GPUMatrix tempGpuVec = matrix_alloc_gpu(b.height,1);
    cusparseSpMatDescr_t spMatrixA,spMatrixA_T;
    status = cusparseCreateCsr(&spMatrixA,A.height,A.width,A.elementSize,A.csrRow,A.csrCol,A.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    const CPUMatrix A_t; /// there is some init here. 
	//init stage
    //beta = norm(b)
    beta = normalVectorNorm(b);
    //u = b/beta
	cudaMemcpy(u.elements, b.elements, b.height*sizeof(double), cudaMemcpyDeviceToDevice);
    scaleNormalvector(u,beta);
    //v = A'*u
    tempDouble = 1; tempDouble2 = 0;
    status = cusparseCsrmvEx_bufferSize(handle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A_T.height,A_T.rows,A_T.elementSize,&tempDouble,CUDA_R_64F,spMatrixA_T,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,u.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&tempInt);
    cudaMalloc(&buffer, tempInt);
    status = cusparseCsrmvEx(handle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A_T.height,A_T.rows,A_T.elementSize,&tempDouble,CUDA_R_64F,spMatrixA,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,u.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&buffer);
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
        status = cusparseCsrmvEx(handle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A.height,A.width,A.elementSize,&tempDouble,CUDA_R_64F,spMatrixA,A.elements,CUDA_R_64F,A.csrRow,A.csrCol,v.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,u.elements,A.height,CUDA_R_64F,&buffer);
        //beta = norm(u);
        beta = normalVectorNorm(u);
        // u = u / beta;
        scaleNormalvector(u,beta);
        // v = A' * u - beta * v;
        tempDouble = 1; tempDouble2 = (-1)*beta;
        status = cusparseCsrmvEx(handle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A_T.height,A_T.width,A_T.elementSize,&tempDouble,CUDA_R_64F,spMatrixA_T,A_T.elements,CUDA_R_64F,A_T.csrRow,A_T.csrCol,u.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&buffer);
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
        //x = (phi / rho) * w + x;          
        vectorAddSub(x,tempVector,true);
        //	w = -(theta / rho) * w + v;
        cudaMemcpy (tempVector.elements, w.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
        scaleNormalvector(w,(theta/rho)); 
        vectorAddSub(w,v,false);
        //check for convergence
        tempDouble = 1; tempDouble2 = (0);
        status = cusparseCsrmvEx(handle,CUSPARSE_ALG_MERGE_PATH,CUSPARSE_OPERATION_NON_TRANSPOSE,A.height,A.width,A.elementSize,&tempDouble,CUDA_R_64F,spMatrixA,A.elements,CUDA_R_64F,A.csrRow,A.csrCol,x.elements,CUDA_R_64F,&tempDouble2,CUDA_R_64F,v.elements,A_T.height,CUDA_R_64F,&buffer);
		//residual = norm(A*x - b);
        //Ax - b (result in tempVector)
        improvment = prev_err-curr_err;
        printf("line: %d size of error: %.6f improvment of: %.6f\n",i,curr_err,improvment);i++;
        if(improvment<ebs) counter++; else counter = 0;
		if(counter>1000) break;
		prev_err = curr_err;
    }
	CPUMatrix result = matrix_alloc_cpu(x.height,x.width);
    matrix_download(x,result);
    cusparseClean(handle,spMatrixA,spMatrixA_T);
	return result;
}

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A, cusparseSpMatDescr_t &A_T){
    cusparseStatus_t status;
    status = cusparseDestroySpMat(A);
    status = cusparseDestroySpMat(A_T);
    status = cusparseDestroy(handle);
    cuSPARSECheck(status,__LINE__);
}


void cuSPARSECheck(cusparseStatus_t status, int line){
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

double normalVectorNorm(GPUMatrix src){
    double res = 0.0; 
    //kernel call here
    return 0.0;
}
void scaleNormalvector(GPUMatrix src,double alpha){
    //kernel call on src vector
}
void vectorAddSub(GPUMatrix a, GPUMatrix b, bool sign){
    GPUMatrix tempGpuVec = matrix_alloc_gpu(a.height,1);
    // some kernel call, result in tempgpu vec
    cudaMemcpy (a.elements, tempGpuVec.elements, a.height*sizeof(double), cudaMemcpyDeviceToDevice);
}




/*
    cusparseAxpby() - for vector addition
    cusparseCsrmvEx() matrix-vector multiplication
    cublas norm
    A is an m×n sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); x and y are vectors;
*/