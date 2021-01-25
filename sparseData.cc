#include "sparseData.h"

/* <<<<<--------- READ DATA IN COMPRESED SPARSE ROW FORMAT ----------->>>>>>>>>> */
CPUMatrix read_matrix_in_csr(const char *path) {
    int height;
    int width;
    double* elements;
    int* csrRow;
    int* csrCol;

    int elementSize = 0;
    int rowSize = 0;
    int columnSize = 0;

    std::fstream file(path);

    if (file.is_open()) {
        int i = 0;
        int j = 0;      /* ElementIdx */

        int numbersInRow = 0;
        int heightIdx = 0;
        int rowIdx = 0;
        int idx = 0;

        double line;
        while (file >> line) {
            if(i == 0) {
                height = (int)line;
                i++;

            }else if (i == 1) {
                width = (int)line;
                i++;

                elementSize = width * height;
                rowSize = width;
                columnSize = width * height;

                elements = new double[elementSize];
                csrRow = new int[rowSize];
                csrCol = new int[columnSize];

            } else {
                if(idx % width == 0) {
                    heightIdx = 0;
                }

                if(idx >= rowSize) {
                    rowSize = 2 * rowSize + 1;
                    csrRow = swapping_i_vector(csrRow, rowSize);
                }

                if (idx % height == 0) {
                    csrRow[rowIdx] = numbersInRow;
                    rowIdx++;
                }

                if (line != 0.0 ) {
                    if(j >= elementSize) {
                        elementSize = 2 * elementSize;
                        elements = swapping_d_vector(elements, elementSize);
                    }
                    elements[j] = line;

                    if(j >= columnSize) {
                        columnSize = 2 * columnSize;
                        csrCol = swapping_i_vector(csrCol, columnSize);
                    }
                    csrCol[j] = heightIdx;
                    
                    numbersInRow++;
                    j++;
                }

                heightIdx++;
                idx++;
            }
        }

        elementSize = j;
        elements = end_d_vector(elements, elementSize);

        columnSize = j;
        csrCol = end_i_vector(csrCol, columnSize);
        
        rowSize = rowIdx + 1;
        csrRow[rowIdx] = numbersInRow;
        csrRow = end_i_vector(csrRow, rowSize);

        file.close();
    } else {
        std::cout << "file not found " << std::endl;
        exit(0);
    }
    
    file.close();

    CPUMatrix matrix = matrix_alloc_sparse_cpu(height, width, elementSize, rowSize, columnSize);
    matrix.elements = elements;
    matrix.csrRow = csrRow;
    matrix.csrCol = csrCol;

    return matrix;

}

CPUMatrix read_data_in_csr(const char *path) {
    int height;
    int width;
    double* elements;
    int *csrIdx;

    int elementSize;
    int csrSize;

    std::fstream file(path);

    if (file.is_open()) {
        int i = 0;
        int j = 0;      /* ElementIdx */

        int idx = 0;

        double line;
        while (file >> line) {
            if(i == 0) {
                height = (int)line;
                i++;

            }else if (i == 1) {
                width = (int)line;
                i++;

                elementSize = width * height;
                csrSize = width;

                elements = new double[elementSize];
                csrIdx = new int[csrSize];

            } else {
                if( line != 0.0) {
                    if(j >= elementSize) {
                        elementSize = 2 * elementSize;
                        elements = swapping_d_vector(elements, elementSize);
                    }
                    elements[j] = line;
                    
                    if(j >= csrSize) {
                        csrSize = 2 * csrSize;
                        csrIdx = swapping_i_vector(csrIdx, csrSize);
                    }
                    csrIdx[j] = idx;
                    j++;
                }
                idx++;
            }
        }

        elementSize = j;
        elements = end_d_vector(elements, elementSize);

        csrSize = j;
        csrIdx = end_i_vector(csrIdx, csrSize);
    } else {
        std::cout << "file not found " << std::endl;
        exit(0);
    }

    for (int idx = 0; idx < elementSize; idx ++) std::cout << elements[idx] << ", ";
    std::cout << std::endl;
    for (int idx = 0; idx < csrSize; idx ++) std::cout << csrIdx[idx] << ", ";

    CPUMatrix matrix = vector_alloc_sparse_cpu(height, width, elementSize, csrSize);
    matrix.elements = elements;
    matrix.csrCol = csrIdx;

    return matrix;
}


double *swapping_d_vector(double *elements, int elementSize) {
    double *tmp = new double[elementSize];
    for (int idx = 0; idx < elementSize / 2; idx ++) tmp[idx] = elements[idx];
    delete[] elements;
    return tmp;
}

int *swapping_i_vector(int *elements, int elementSize) {
    int *tmp = new int[elementSize];
    for (int idx = 0; idx < elementSize / 2; idx ++) tmp[idx] = elements[idx];
    delete[] elements;
    return tmp;
}

double *end_d_vector(double *elements, int size) {
    double *tmp = new double[size];
    for (int idx = 0; idx < size; idx ++) tmp[idx] = elements[idx];
    return tmp;
}

int *end_i_vector(int *elements, int size) {
    int *tmp = new int[size];
    for (int idx = 0; idx < size; idx ++) tmp[idx] = elements[idx];
    return tmp;
}