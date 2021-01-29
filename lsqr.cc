#include <iostream>
#include <cuda_runtime.h>
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
    //std::tuple<int, int , double*> A = read_file(pathMatrixA);
    std::tuple<int, int , double*> b = read_file(pathVectorb);


    /* <<<< ---------------- READ DATA IN CPUMATRIX ----------------------- >>>> */
    //CPUMatrix cpuMatrixA = matrix_alloc_cpu(std::get<0>(A), std::get<1>(A));
    //cpuMatrixA.elements = std::get<2>(A);

    CPUMatrix cpuVector_b = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    cpuVector_b.elements = std::get<2>(b);

    
    /* <<<< ---------------- CALCULATE LSQR ONLY WITH cuBLAS-LIBARY ----------------------- >>>> */



    //* <<<< ---------------- CALCULATE LSQR ONLY WITH cuSPARSE LIBARY ----------------------- >>>> */
    


    /* <<<< ---------------- CALCULATE LSQR ONLY WITH KERNELS ----------------------- >>>> */
    CPUMatrix resultKernel = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    
    print_matrix_vector_dense_format(cpuVector_b.elements, cpuVector_b.width * cpuVector_b.height);
    resultKernel = sparseLSQR_with_kernels(cpuVector_b, cpuVector_b, 0.0, ebs);
    
    // Testing
    //compare_lsqr(cpuMatrixA, cpuVectorb, cpuVectorb, lambda, ebs);



    /* <<<< ---------------- FREE CPUMATRIX MEMORY ----------------------- >>>>  */
    //free(std::get<2>(A));
    free(std::get<2>(b));

}