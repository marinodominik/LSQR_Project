#include "matrix.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>

CPUMatrix matrix_alloc_cpu(int height, int width) {
    CPUMatrix m;
	m.height = height;
    m.width = width;
    m.elements = new double[width * height];
    return m;
}

CPUMatrix matrix_alloc_sparse_cpu(int height, int width, int elementSize, int rowSize, int columnSize) {
	CPUMatrix m;
	m.height = height;
	m.width = width;
	m.elements = new double[elementSize];

	m.rowSize = rowSize;
	m.csrRow = new int[rowSize];

	m.columnSize = columnSize;
	m.csrCol = new int[columnSize];
	return m;
}

CPUMatrix vector_alloc_sparse_cpu(int height, int width, int sizeElements, int colSize) {
	CPUMatrix m;
	m.height = height;
	m.width = width;
	m.elements = new double[sizeElements];

	m.columnSize = colSize;
	m.csrCol = new int[colSize];

	return m;
}


void matrix_free_cpu(CPUMatrix &m) {
    delete[] m.elements; 
}

void matrix_free_sparse_cpu(CPUMatrix &m) {
	delete[] m.elements;
	delete[] m.csrRow;
	delete[] m.csrCol;
}



GPUMatrix matrix_alloc_gpu(int height, int width) {
    GPUMatrix Md;
	Md.height = height;
	Md.width = width;
	int size = width * height * sizeof(double);
	cudaError_t err = cudaMallocPitch(&Md.elements, &Md.pitch,width*sizeof(float),height);
	return Md;
}


void matrix_free_gpu(GPUMatrix &m) {
    cudaFree(m.elements);
}


void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	int size = src.height*src.width*sizeof(double);
	cudaMemcpy(dst.elements, src.elements, size, cudaMemcpyHostToDevice);
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	int size = src.height*src.width*sizeof(double);
	cudaMemcpy(dst.elements, src.elements, size, cudaMemcpyDeviceToHost);
}

