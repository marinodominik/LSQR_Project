#include "lsqrCUDAcuSparse.h"
void printVectorj(int iteration,GPUMatrix x, const char* name);
CPUMatrix cusparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs){
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cuSPARSECheck(__LINE__);
    GPUMatrix u = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix v = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix w = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix x = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix GPUb = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix tempVector = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix tempVector2 = matrix_alloc_gpu(b.height,b.width); 
    cuSPARSECheck(__LINE__);
    matrix_upload(b,GPUb);
    CPUMatrix res = cusparseLSQR_aux(A,GPUb,u,v,w,x,tempVector,tempVector2,ebs);
    return res; 
}

CPUMatrix cusparseLSQR_aux(const CPUMatrix &A, const GPUMatrix &VECb,GPUMatrix &VECu,GPUMatrix &VECv,GPUMatrix &VECw,GPUMatrix &VECx,GPUMatrix &tempVector,GPUMatrix &tempVector2,double ebs){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err,prev_err,improvment;
    size_t tempInt;
    double *buffer;
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cuSPARSECheck(__LINE__);
    prev_err = 100000000; 
    cusparseSpMatDescr_t spMatrixA;
    cusparseDnVecDescr_t b,u,v,w,x,tempDense,tempDense2;
    GPUMatrix GPUA =  matrix_alloc_sparse_gpu(A.height,A.width,A.elementSize,A.rowSize,A.columnSize);
    matrix_upload_cuSparse(A,GPUA);
    cuSPARSECheck(__LINE__);
    cusparseCreateCsr(&spMatrixA,GPUA.height,GPUA.width,GPUA.elementSize,GPUA.csrRow,GPUA.csrCol,GPUA.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cuSPARSECheck(__LINE__);
    cusparseCreateDnVec(&b,VECb.height,VECb.elements,CUDA_R_64F);
    cusparseCreateDnVec(&u,VECb.height,VECu.elements,CUDA_R_64F);
    cusparseCreateDnVec(&v,VECb.height,VECv.elements,CUDA_R_64F);
    cusparseCreateDnVec(&w,VECb.height,VECw.elements,CUDA_R_64F);
    cusparseCreateDnVec(&x,VECb.height,VECx.elements,CUDA_R_64F);
    cusparseCreateDnVec(&tempDense,VECb.height,tempVector.elements,CUDA_R_64F);
    cusparseCreateDnVec(&tempDense2,VECb.height,tempVector2.elements,CUDA_R_64F);
    cuSPARSECheck(__LINE__);
	//init stage
    //beta = norm(b)
    beta = normalVectorNorm(b,tempVector);
    //u = b/beta
    cusparseDnVecGetValues(b,(void**)&tempVector.elements);   
	cudaMemcpy (VECu.elements,tempVector.elements, VECu.height*sizeof(double), cudaMemcpyDeviceToDevice);
    scaleNormalvector(u,1/beta,tempVector);
    cuSPARSECheck(__LINE__);
    printDenseVector(u,"u",tempVector);
    cuSPARSECheck(__LINE__);
   // printSparseMatrix(spMatrixA,"A",tempVector);
    cuSPARSECheck(__LINE__);
    //v = A'*u
    tempDouble = 1; tempDouble2 = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&tempInt);
    cuSPARSECheck(__LINE__);
    cudaMalloc(&buffer, tempInt);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
    cuSPARSECheck(__LINE__);

    //alpha = norm(v)
    alpha = normalVectorNorm(v,tempVector);
    cuSPARSECheck(__LINE__);

    //v = v/alpha;
    scaleNormalvector(v,1/alpha,tempVector);
    cuSPARSECheck(__LINE__);

    printDenseVector(v,"v",tempVector);
    //w = v;
    cusparseDnVecGetValues(v,(void**)&tempVector.elements);   
    cudaMemcpy (VECw.elements,tempVector.elements, VECv.height*sizeof(double), cudaMemcpyDeviceToDevice);
    cuSPARSECheck(__LINE__);

    //printDenseVector(w,"w",tempVector);
    
    phi_tag = beta; rho_tag = alpha;
	int i = 0;
	while(true){
		//next bidiagonlization
        // u = A * v - alpha * u;
        tempDouble = 1; tempDouble2 = (-1)*alpha;
        cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&tempDouble,spMatrixA,v,&tempDouble2,u,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        cuSPARSECheck(__LINE__);
        //beta = norm(u);
        beta = normalVectorNorm(u,tempVector);
        cuSPARSECheck(__LINE__);
        // u = u / beta;
        scaleNormalvector(u,1/beta,tempVector);
        cuSPARSECheck(__LINE__);
        printDenseVector(u,"u",tempVector);

        // v = A' * u - beta * v;
        tempDouble = 1; tempDouble2 = (-1)*beta;
        cusparseSpMV(handle,CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        cuSPARSECheck(__LINE__);
        //alpha = norm(v)
        alpha = normalVectorNorm(v,tempVector);
        cuSPARSECheck(__LINE__);
        //v = v/alpha;
        scaleNormalvector(v,1/alpha,tempVector);
        cuSPARSECheck(__LINE__);
        printDenseVector(v,"v",tempVector);
		//next orthogonal transformation
		rho = sqrt(pow (rho_tag, 2.0) + pow (beta, 2.0));
		c = rho_tag / rho;
		s = beta / rho;
		theta = s * alpha;
		rho_tag = (-1) * c * alpha;
		phi = c * phi_tag;
		phi_tag = s * phi_tag;
        printf("constants: alpha: %.6f beta:%.6f\n",alpha,beta);
		printf("constants: rho: %.6f c: %.6f s: %.6f theta: %.6f rho_tag: %.6f phi: %.6f\n phi_tag: %.6f\n",rho,c,s,theta,rho_tag,phi,phi_tag);
        //updating x,w
        printDenseVector(w,"w",tempVector);
        copyVector(tempDense,w,tempVector);
        scaleNormalvector(tempDense,phi/rho,tempVector); 
        cuSPARSECheck(__LINE__);
        //x = x + (phi / rho) * w ;          
        vectorAddSub(x,tempDense,true,tempVector);
        cuSPARSECheck(__LINE__);
        printDenseVector(x,"x",tempVector);
        //	w = -(theta / rho) * w + v;
        scaleNormalvector(w,(theta/rho)*(-1),tempVector); 
        cuSPARSECheck(__LINE__);
        vectorAddSub(w,v,true,tempVector);
        cuSPARSECheck(__LINE__);
        //printDenseVector(w,"w",tempVector);
        printDenseVector(b,"b",tempVector);

        //check for convergence
        tempDouble = 1; tempDouble2 = (-1);
        cusparseDnVecGetValues(b,(void**)&tempVector.elements); 
        cuSPARSECheck(__LINE__);
        cudaMemcpy (tempVector2.elements,tempVector.elements, VECu.height*sizeof(double), cudaMemcpyDeviceToDevice);
        cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&tempDouble,spMatrixA,x,&tempDouble2,tempDense2,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        cuSPARSECheck(__LINE__);
        //residual = norm(A*x - b);
        //Ax - b (result in tempDense2)
        printDenseVector(tempDense2,"result",tempVector);
        curr_err = normalVectorNorm(tempDense2,tempVector2);
        printDenseVector(tempDense2,"result",tempVector);

        cuSPARSECheck(__LINE__);
        printDenseVector(tempDense,"temo",tempVector);

        improvment = prev_err-curr_err;
        printf("line: %d size of error: %.6f improvment of: %.6f\n",i,curr_err,improvment);i++;
        if(i==A.height) break;
        prev_err = curr_err;
    }
    printf("LSQR using cuSPARSE finished.\n Iterations num: %d\n Size of error: %.6f\n",i,curr_err);
    CPUMatrix result = matrix_alloc_cpu(VECb.height,VECb.width);
    cusparseDnVecGetValues(x,(void**)&tempVector.elements);
    matrix_download(tempVector,result);
    cusparseClean(handle,spMatrixA);
	return result;
}

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A){
    cusparseDestroySpMat(A);
    cusparseDestroy(handle);
    cuSPARSECheck(__LINE__);
}


void cuSPARSECheck(int line){
    const cudaError_t err = cudaGetLastError();                            
	if (err != cudaSuccess) {                                              
    	const char *const err_str = cudaGetErrorString(err);               
    	std::cerr << "Cuda error in " << __FILE__ << ":" << line - 1   
            << ": " << err_str << " (" << err << ")" << std::endl;   
            exit(EXIT_FAILURE);                                                                    
	}
}

double normalVectorNorm(cusparseDnVecDescr_t src, GPUMatrix temp){
    cusparseDnVecGetValues(src,(void**)&temp.elements);
    printVector(-1,temp,"norm");
    cuSPARSECheck(__LINE__);
    double res = getNorm2(temp);
    printf("res:  %lf ", res);
    return res;
}
void scaleNormalvector(cusparseDnVecDescr_t src,double alpha,GPUMatrix temp){
    cusparseDnVecGetValues(src,(void**)&temp.elements);
    GPUMatrix res = multiply_scalar_vector(temp,alpha);
    cusparseDnVecSetValues(src,res.elements);
}
void vectorAddSub(cusparseDnVecDescr_t a, cusparseDnVecDescr_t b, bool sign,GPUMatrix temp){  // result overrides to a
    GPUMatrix temp2 = matrix_alloc_gpu(temp.height,temp.width); 
    cusparseDnVecGetValues(a,(void**)&temp.elements);
    cusparseDnVecGetValues(b,(void**)&temp2.elements);   
    GPUMatrix res = get_add_subtract_vector(temp,temp2,sign);
    cusparseDnVecSetValues(a,res.elements);
    matrix_free_gpu(temp2);
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
    matrix_free_cpu(tempCPUMatrix);
}
void printSparseMatrix(cusparseSpMatDescr_t src,const char* name,GPUMatrix temp){
    GPUMatrix tempGPU = matrix_alloc_gpu(temp.height,temp.height);
    cusparseSpMatGetValues(src,(void**)&tempGPU.elements);
    printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(tempGPU.height,tempGPU.height);
	matrix_download(tempGPU,tempCPUMatrix);
	for(int i = 0; i < tempCPUMatrix.height*tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
    printf("\n");
    matrix_free_gpu(tempGPU);
    matrix_free_cpu(tempCPUMatrix);
}
void printNormalVector(GPUMatrix x, const char* name){
    printf("%s: \n",name);
    CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height,1);
    matrix_download_cuSparse(x,tempCPUMatrix);
	for(int i = 0; i < x.elementSize; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
    printf("\n");
    matrix_free_cpu(tempCPUMatrix);
}
void printVectorj(int iteration,GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height,x.width);
	matrix_download(x,tempCPUMatrix);
	//printf("iteration number: %d\n", iteration);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}