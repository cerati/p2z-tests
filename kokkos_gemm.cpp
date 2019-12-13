
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// kokkos
#include <Kokkos_Core.hpp>
#include "kokkos_config.h"
#include "kokkos_gemm.h"

#define AVAL 5.
#define BVAL 6.
#define TOL 0.0001


void run_normal_test(int N, int M, int nrepeat) {
  
  // Allocate y, x vectors and Matrix A on device.
  // ViewMatrixType A( "A", N, M );
  // ViewMatrixType B( "B", M, N );
  // ViewMatrixType C( "C", N, N );
  ViewMatrixTypeL A( "A", N, M );
  ViewMatrixTypeR B( "B", M, N );
  ViewMatrixTypeL C( "C", N, N );
  init_matrices(A, B, C, N, M);

  // Timer products.
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {

    gemm(A, B, C, N, M);

  }

  gettimeofday( &end, NULL );

  // Calculate time.
  double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N*N ) );

  double err = check_result(C, N, M);
  if (err > TOL)  {
    printf("ERROR in computation; result check failed; error = %f\n", err);
  }
  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s )\n",
          N, M, nrepeat, Gbytes * 1000, time);

}

void init_matrices(ViewMatrixTypeL A, ViewMatrixTypeR B, ViewMatrixTypeL C, int N, int M) {

  // Initialize A matrix on host.
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < M; ++i ) {
      A( j, i ) = AVAL;
    }
  }
  // Initialize B matrix on host.
  for ( int j = 0; j <M; ++j ) {
    for ( int i = 0; i < N; ++i ) {
      B( j, i ) = BVAL;
    }
  }
  // Initialize C matrix on host.
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < N; ++i ) {
      C( j, i ) = 1;
    }
  }

}


void gemm(ViewMatrixTypeL A, ViewMatrixTypeR B, ViewMatrixTypeL C, int N, int M) {

  Kokkos::parallel_for( N, KOKKOS_LAMBDA ( int i ) {
  // for ( int i = 0; i < N; ++i ) {
    for ( int j = 0; j < N; ++j ) {
      C(i,j) = 0.0;
      for ( int k = 0; k < M; ++k ) {
        C(i,j) += A(i,k) * B(k,j);
      }
    }
  // }
  });

}


double check_result(ViewMatrixTypeL C, int N, int M) {

  int i, j;

  double e  = 0.0;
  double ee = 0.0;
  double v  = AVAL * BVAL * M;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      e = C(i,j) - v;
      ee += e * e;
    }
  }

  return ee;
}



void run_batch_test(int N, int M, int nrepeat, int nbatches, int batch_size) {
  
  // Allocate y, x vectors and Matrix A on device.
  ViewMatrixBatch A( "A", nbatches, N, M, batch_size );
  ViewMatrixBatch B( "B", nbatches, M, N, batch_size );
  ViewMatrixBatch C( "C", nbatches, N, N, batch_size );
  init_batches(A, B, C, N, M, nbatches, batch_size);

  // Timer products.
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {

    batch_gemm(A, B, C, N, M, nbatches, batch_size);

  }

  gettimeofday( &end, NULL );

  // Calculate time.
  double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N*N ) );

  double err = check_result_batch(C, N, M, nbatches, batch_size);
  if (err > TOL)  {
    printf("ERROR in computation; result check failed; error = %f\n", err);
  }
  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s )\n",
          N, M, nrepeat, Gbytes * 1000, time);

}


void init_batches(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches, int batch_size) {

  for (int batch = 0; batch < nbatches; batch++) {

    // Initialize A matrix on host.
    for ( int j = 0; j < N; ++j ) {
      for ( int i = 0; i < M; ++i ) {
        for ( int b = 0; b < batch_size; ++b ) {
          A(batch, j, i, b) = AVAL;
        }
      }
    }

    // Initialize B matrix on host.
    for ( int j = 0; j <M; ++j ) {
      for ( int i = 0; i < N; ++i ) {
        for ( int b = 0; b < batch_size; ++b ) {
          B(batch, j, i, b) = BVAL;
        }
      }
    }

    // Initialize C matrix on host.
    for ( int j = 0; j < N; ++j ) {
      for ( int i = 0; i < N; ++i ) {
        for ( int b = 0; b < batch_size; ++b ) {
          C(batch, j, i, b) = 1;
        }
      }
    }

  } // batch loop

}

void batch_gemm(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches, int batch_size) {
  
  Kokkos::parallel_for( nbatches, KOKKOS_LAMBDA ( int batch ) {
  for ( int i = 0; i < N; ++i ) {
    for ( int j = 0; j < N; ++j ) {

      for ( int b = 0; b < batch_size; ++b ) 
        C(batch,i,j,b) = 0.0;

      for ( int k = 0; k < M; ++k ) {
        for ( int b = 0; b < batch_size; ++b ) {
          C(batch,i,j,b) += A(batch,i,k,b) * B(batch,k,j,b);
        }
      }
      
    }
  }
  });

}


double check_result_batch(ViewMatrixBatch C, int N, int M, int nbatches, int batch_size) {

  double e  = 0.0;
  double ee = 0.0;
  double v  = AVAL * BVAL * M;

  for (int batch = 0; batch < nbatches; ++batch) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for ( int b = 0; b < batch_size; ++b ) {
        e = C(batch,i,j,b) - v;
        ee += e * e;
      }
    }
  }
  }

  return ee;
}







