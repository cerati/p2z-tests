#ifndef __GEMM_KOKKOS__
#define __GEMM_KOKKOS__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// kokkos
#include <Kokkos_Core.hpp>
#include "kokkos_config.h"

void run_batch_test(int N, int M, int nrepeat, int nbatches, int batch_size);
void init_batches(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches, int batch_size);
void batch_gemm(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches, int batch_size);
double check_result_batch(ViewMatrixBatch C, int N, int M, int nbatches, int batch_size);

void run_normal_test(int N, int M, int nrepeat);
void init_matrices(ViewMatrixTypeL A, ViewMatrixTypeR B, ViewMatrixTypeL C, int N, int M);
void gemm(ViewMatrixTypeL A, ViewMatrixTypeR B, ViewMatrixTypeL C, int N, int M);
double check_result(ViewMatrixTypeL C, int N, int M);

#endif