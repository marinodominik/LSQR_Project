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
    cusparseSpMatDescr_t spMatrixA;
    cusparseSpVecDescr_t spVectorb;
    cusparseDnVecDescr_t u,v,w,x,tempVector;

    cusparseCreateCsr(&A,A.height,A.rows,A.elementSize,A.csrRow,A.csrCol,A.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cusparseCreateSpVec(&spVectorb,b.height,b.elementSize,b.csrCol,b.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    
    double *zeros = new double[b.elementSize]; //to init helper vectors
    
    cusparseCreateDnVec(&u,b.height,b.elementSize,b.csrCol,zeros,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cusparseCreateDnVec(&v,b.height,b.elementSize,b.csrCol,zeros,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cusparseCreateDnVec(&w,b.height,b.elementSize,b.csrCol,zeros,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cusparseCreateDnVec(&x,b.height,b.elementSize,b.csrCol,zeros,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    
    cusparseDestroy(handle);
    return sparseLSQR_aux();
}

CPUMatrix sparseLSQR_aux(const cusparseSpMatDescr_t &A, const cusparseSpVecDescr_t &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__line)
	prev_err = 100000000; 
	//init stage
	//beta = norm(b)
	//u = b/beta
	//v = A'*u
	//alpha = norm(v)
	//v = v/alpha;
    //w = v;
    //phi_hat = beta;
    //rho_hat = alpha;

	int i = 0, counter = 0;
	while(true){
		//next bidiagonlization
		// u = A * v - alpha * u;
		//beta = norm(u);
		// u = u / beta;
		// v = A' * u - beta * v;
		//alpha = norm(v)
		//v = v/alpha;
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
		//	w = v - (theta / rho) * w ;
		//check for convergence
		//residual = norm(A*x - b);
        //Ax - b (result in tempVector)
    }

}

void cuSPARSECheck(cusparseStatus_t status, int line){
	if(status != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

double sparseVectorNorm(cusparseSpVecDescr_t vector){
    return 0.0;
}
void scaleVector(cusparseSpVecDescr_t vector){

}


/*
    cusparseAxpby() - for vector addition
    cusparseCsrmvEx() matrix-vector multiplication
    cublas norm

    A is an m×n sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); x and y are vectors;

*/