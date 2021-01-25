#include "helper.h"

double** reshape_array_1D_To_2D(double *arr, int height, int width) {
    double** reshaped = 0;
    reshaped = new double*[height];

    for(int i = 0; i < height; i++) {
        reshaped[i] = new double[width];
        for (int j = 0; j < width; j++) {
            reshaped[i][j] = arr[i * width + j];
        }
    }
    return reshaped;
}

void print_2D_array(double **arr, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << arr[i][j]<< " ";
        }
        std::cout << std::endl;
    }
}

void print_matrix_vector_dense_format(double *elements, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << elements[i] << " ";
    }
    std::cout << std::endl;
}

void print_matrix_vector_dense_format(int* elements, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << elements[i] << " ";
    }
    std::cout << std::endl;
}

std::tuple<int, int , double*> read_file(const char* path) {
   /* Function for reading data from a file
    First value of the file have to be the numbers of the rows and the second number is 
    the column number of the matrix/vector. Everything else numbers for the Matrix/Vector.
    :param: path: location of the file
    return:
    :param: height of the matrix/vector
    :param: width of the matrix/vector
    :param: Matrix/Vector array as a 1D-Array
    */

   int height;
   int width;
   double* elements;
    
    std::fstream file(path);

    if (file.is_open()) {
        int i = 0;
        int j = 0;

        double line;
        while (file >> line) {
            if (i == 0) {
                height = (int)line;
                i++;
            } else if (i == 1) {
                width = (int)line;
                i++;
            } else {
                /* <<<< -------   read Matrix/Vector  ------- >>>>> */
                if (i == 2 && j == 0) {
                    elements = new double[width * height];
                    elements[j] = line;
                    i++;
                    j++;
                } else {
                    elements[j] = line;
                    j++;
                }
            }
        }
        file.close();
    } else {
        std::cout << "file not found " << std::endl;
        exit(0);
    }

   return std::make_tuple(height, width, elements);
}

void save_file(const char* path, double* elements, int height, int width) {
    std::ofstream output;
    output.open (path);
    int j = 0;
    for(int i=0; i<height + 2;i++){
        if (i == 0) {
            output << height << std::endl;
        }else if (i == 1) {
            output << width << std::endl;
        } else {
            output << elements[j] << std::endl;
            j++;
        }
    }
    output.close();
}

bool check_hardware() {

    //TODO
}


inline unsigned int div_up(unsigned int numerator, unsigned int denominator) //numerator = zähler, denumerator = nenner
{
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}