#pragma once
#include "matrix.h"


void lsqr(const char *pathMatrixA, const char *pathVector_b, double lambda);
void printTruncatedVector(CPUMatrix toPrint);