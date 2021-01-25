#include <iostream>
#include <cuda_runtime.h>
#include "matrix.h"
#include "lsqr.h"
#include "lsqrCUDAcuBlas.h"
#include "lsqrCUDAcuSparse.h"
#include "testing.h"
#include "helper.h"
#include "sparseData.h"


void lsqr(const char *pathMatrixA, const char *pathVectorb, double lambda) {
    double ebs = 1e-9;
    
    //print_matrix_vector_dense_format(A.elements, A.width * A.height);
    //print_matrix_vector_dense_format(A.csrRow, A.rowSize);
    //print_matrix_vector_dense_format(A.csrCol, A.columnSize);

    //std::tuple<int, int , double*> A = read_file(pathMatrixA);
    //std::tuple<int, int , double*> b = read_file(pathVectorb);

    //CPUMatrix cpuMatrixA = matrix_alloc_cpu(std::get<0>(A), std::get<1>(A));
    //cpuMatrixA.elements = std::get<2>(A);

    //CPUMatrix cpuVectorb = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    //cpuVectorb.elements = std::get<2>(b);

    //CPUMatrix result = normalLSQR(cpuMatrixA,cpuVectorb,ebs);

    //compare_lsqr(cpuMatrixA, cpuVectorb, cpuVectorb, lambda, ebs);
}