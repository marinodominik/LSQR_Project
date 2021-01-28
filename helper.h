#pragma once
#include <iostream>
#include <tuple>
#include <fstream>

double** reshape_array_1D_To_2D(double *arr, int width, int height);

void print_2D_array(double **arr, int width, int height);

void print_matrix_vector_dense_format(double *elements, int size);
void print_matrix_vector_dense_format(int* elements, int size);

std::tuple<int, int , double*> read_file(const char* path);

void save_file(const char* path, double* elements, int height, int widht);
