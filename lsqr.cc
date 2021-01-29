#include <iostream>
#include <cuda_runtime.h>
#include <tuple>

#include "matrix.h"
#include "lsqr.h"
#include "lsqrCUDAcuBlas.h"
#include "lsqrCUDAcuSparse.h"
#include "lsqrCUDAcuSparseKernel.h"
#include "testing.h"
#include "helper.h"
#include "sparseData.h"


void lsqr(const char *pathMatrixA, const char *pathVectorb, double lambda) {
    double ebs = 1e-9;
    
    /* <<<< ---------------------- READ DATA ----------------------------- >>>> */
    std::tuple<int, int , double*> A = read_file(pathMatrixA);
    std::tuple<int, int , double*> b = read_file(pathVectorb);


    /* <<<< ---------------- READ DATA IN CPUMATRIX ----------------------- >>>> */
    CPUMatrix cpuMatrixA = matrix_alloc_cpu(std::get<0>(A), std::get<1>(A));
    cpuMatrixA.elements = std::get<2>(A);

    CPUMatrix cpuVector_b = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    cpuVector_b.elements = std::get<2>(b);

    
    /* <<<< ---------------- CALCULATE LSQR ONLY WITH cuBLAS-LIBARY ----------------------- >>>> */
    std::cout << "Starting LSQR using cuBLAS-LIBARY\n" << std::endl;
    CPUMatrix cuBLASResult = cublasLSQR(cpuMatrixA,cpuVector_b,ebs);
    printTruncatedVector(cuBLASResult);


    //* <<<< ---------------- CALCULATE LSQR ONLY WITH cuSPARSE LIBARY ----------------------- >>>> */
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
=======
>>>>>>> 3e974da87ea89eed9325cafd0d570957000226a1
    std::cout << "Starting LSQR using cuSPAPRSE-LIBARY\n" << std::endl;
    CPUMatrix sparseMatrixA = read_matrix_in_csr(pathMatrixA);
    CPUMatrix cuSPARSEResult = cusparseLSQR(sparseMatrixA,cpuVector_b,ebs);
    printTruncatedVector(cuSPARSEResult);
<<<<<<< HEAD
=======
    
>>>>>>> 7a1a13c0c45aa6abb7468fec5aae88197943e457
=======
>>>>>>> 4657c19d125bb520bff5b7f57d425306cc30c178
>>>>>>> 3e974da87ea89eed9325cafd0d570957000226a1


    /* <<<< ---------------- CALCULATE LSQR ONLY WITH KERNELS ----------------------- >>>> */
    std::cout << "Starting LSQR using kernels\n" << std::endl;
    CPUMatrix resultKernel = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    
    print_matrix_vector_dense_format(cpuVector_b.elements, cpuVector_b.width * cpuVector_b.height);
    resultKernel = sparseLSQR_with_kernels(cpuVector_b, cpuVector_b, 0.0, ebs);
    
    // Testing
    //compare_lsqr(cpuMatrixA, cpuVectorb, cpuVectorb, lambda, ebs);



    /* <<<< ---------------- FREE CPUMATRIX MEMORY ----------------------- >>>>  */
    //free(std::get<2>(A));
    free(std::get<2>(b));

}

void printTruncatedVector(CPUMatrix toPrint){
    std::cout<<"[";
    for(int i=0; i<10; i++){
        if(i==toPrint.height){
            std::cout<<"]\n";
            return;
        }
        if(i>0) std::cout<<",";
        std::cout<<toPrint.elements[i]; 
    }
    std::cout<<"...]\n"<<std::endl;
}