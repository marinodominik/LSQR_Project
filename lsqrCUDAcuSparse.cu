#include "lsqrCUDAcuSparse.h"

CPUMatrix cusparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs){
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__LINE__);
    GPUMatrix u = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix v = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix w = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix x = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix GPUb = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix tempVector = matrix_alloc_gpu(b.height,b.width); 
    matrix_upload(b,GPUb);
    CPUMatrix res = cusparseLSQR_aux(A,GPUb,u,v,w,x,tempVector,ebs);
    return res; 
}

CPUMatrix cusparseLSQR_aux(const CPUMatrix &A, const GPUMatrix &VECb,GPUMatrix &VECu,GPUMatrix &VECv,GPUMatrix &VECw,GPUMatrix &VECx,GPUMatrix &tempVector,double ebs){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
    size_t tempInt;
    double *buffer;
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    cuSPARSECheck(status,__LINE__);
    prev_err = 100000000; 
    cusparseSpMatDescr_t spMatrixA;
    cusparseDnVecDescr_t b,u,v,w,x,tempDense;
    status = cusparseCreateCsr(&spMatrixA,A.height,A.width,A.elementSize,A.csrRow,A.csrCol,A.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cusparseCreateDnVec(&b,VECb.height,VECb.elements,CUDA_R_64F);
    cusparseCreateDnVec(&u,VECb.height,VECu.elements,CUDA_R_64F);
    cusparseCreateDnVec(&v,VECb.height,VECv.elements,CUDA_R_64F);
    cusparseCreateDnVec(&w,VECb.height,VECw.elements,CUDA_R_64F);
    cusparseCreateDnVec(&x,VECb.height,VECx.elements,CUDA_R_64F);
    cusparseCreateDnVec(&tempDense,VECb.height,tempVector.elements,CUDA_R_64F);

	//init stage
    //beta = norm(b)
    beta = normalVectorNorm(b,tempVector);
    beta =2;//TODO!!
    //u = b/beta
    copyVector(u,b,tempVector);
    printDenseVector(u,"u",tempVector);
    scaleNormalvector(u,1/beta,tempVector);
    printDenseVector(u,"u2",tempVector);
    //v = A'*u
    tempDouble = 1; tempDouble2 = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&tempInt);
    cudaMalloc(&buffer, tempInt);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
    //alpha = norm(v)
    alpha = normalVectorNorm(v,tempVector);
    //v = v/alpha;
    scaleNormalvector(v,1/alpha,tempVector);
    //w = v;
    copyVector(w,v,tempVector);
    phi_tag = beta; rho_tag = alpha;
	int i = 0, counter = 0;
	while(true){
		//next bidiagonlization
        // u = A * v - alpha * u;
        tempDouble = 1; tempDouble2 = (-1)*alpha;
        cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&tempDouble,spMatrixA,v,&tempDouble2,u,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        //beta = norm(u);
        beta = normalVectorNorm(u,tempVector);
        // u = u / beta;
        scaleNormalvector(u,beta,tempVector);
        // v = A' * u - beta * v;
        tempDouble = 1; tempDouble2 = (-1)*beta;
        cusparseSpMV(handle,CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        //alpha = norm(v)
        alpha = normalVectorNorm(v,tempVector);
        //v = v/alpha;
        scaleNormalvector(v,1/alpha,tempVector);
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
        copyVector(tempDense,w,tempVector);
        scaleNormalvector(tempDense,phi/rho,tempVector); 
        //x = x + (phi / rho) * w ;          
        vectorAddSub(x,tempDense,true,tempVector);
        //	w = -(theta / rho) * w + v;
        scaleNormalvector(w,(theta/rho)*(-1),tempVector); 
        vectorAddSub(w,v,true,tempVector);
        //check for convergence
        tempDouble = 1; tempDouble2 = (-1);
        copyVector(tempDense,b,tempVector);
        cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&tempDouble,spMatrixA,x,&tempDouble2,tempDense,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
		//residual = norm(A*x - b);
        //Ax - b (result in tempDense)
        curr_err = normalVectorNorm(tempDense,tempVector);
        improvment = prev_err-curr_err;
        printf("line: %d size of error: %.6f improvment of: %.6f\n",i,curr_err,improvment);i++;
        if(i==A.height) break;
    }
    CPUMatrix result = matrix_alloc_cpu(VECb.height,VECb.width);
    cusparseDnVecGetValues(x,(void**)&tempVector.elements);
    matrix_download(tempVector,result);
    cusparseClean(handle,spMatrixA);
	return result;
}

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A){
    cusparseStatus_t status;
    status = cusparseDestroySpMat(A);
    status = cusparseDestroy(handle);
    cuSPARSECheck(status,__LINE__);
}


void cuSPARSECheck(cusparseStatus_t status, int line){
	if(status != CUSPARSE_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

double normalVectorNorm(cusparseDnVecDescr_t src, GPUMatrix temp){
    cusparseDnVecGetValues(src,(void**)&temp.elements);
    return getNorm2(temp);
}
void scaleNormalvector(cusparseDnVecDescr_t src,double alpha,GPUMatrix temp){
    cusparseDnVecGetValues(src,(void**)&temp.elements);
    printNormalVector(temp, "temp in scale");
    GPUMatrix res = multiply_scalar_vector(temp,alpha);
    printNormalVector(res, "Res in scale");
    cusparseDnVecSetValues(src,res.elements);
}
void vectorAddSub(cusparseDnVecDescr_t a, cusparseDnVecDescr_t b, bool sign,GPUMatrix temp){  // result overrides to a
    GPUMatrix temp2 = matrix_alloc_gpu(temp.height,temp.width); 
    cusparseDnVecGetValues(a,(void**)&temp.elements);
    cusparseDnVecGetValues(b,(void**)&temp2.elements);   
    GPUMatrix res = get_add_subtract_vector(temp,temp2,sign);
    cusparseDnVecSetValues(a,res.elements);
}
void copyVector(cusparseDnVecDescr_t dst,cusparseDnVecDescr_t src,GPUMatrix temp){
    cusparseDnVecGetValues(src,(void**)&temp.elements);
    cusparseDnVecSetValues(dst,temp.elements);
}

void printDenseVector(cusparseDnVecDescr_t src,const char* name,GPUMatrix temp){
    cusparseDnVecGetValues(src,(void**)&temp.elements);
    printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(temp.height,temp.width);
	matrix_download(temp,tempCPUMatrix);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}
void printNormalVector(GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height,x.width);
	matrix_download(x,tempCPUMatrix);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}