
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// kokkos
#include <Kokkos_Core.hpp>
// our typedefs curtisy of the examples
#include "kokkos_config.h"

#define AVAL 5.
#define BVAL 6.
#define TOL 0.0001

void run_batch_test(int N, int M, int nrepeat, int nbatches);
void init_batches(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches);
void batch_gemm(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches);
double check_result_batch(ViewMatrixBatch C, int N, int M, int nbatches);

void run_normal_test(int N, int M, int nrepeat);
void init_matrices(ViewMatrixTypeL A, ViewMatrixTypeR B, ViewMatrixTypeL C, int N, int M);
void gemm(ViewMatrixTypeL A, ViewMatrixTypeR B, ViewMatrixTypeL C, int N, int M);
void read_input(int argc, char* argv[], int &N, int &M, int &S, int &nrepeat, int &batch_size, int &nbatches, bool &use_batches);
double check_result(ViewMatrixTypeL C, int N, int M);


int main( int argc, char* argv[] )
{
  int N = -1;         // number of rows 1024
  int M = -1;         // number of columns 1024
  int S = -1;         // total size 1024*1024
  int nrepeat = 100;  // number of repeats of the test
  int nbatches = 100;  // number of batches
  int batch_size = 100;  // number of mats per batch
  bool use_batches = false;

  read_input(argc, argv, N, M, S, nrepeat, batch_size, nbatches, use_batches);

  Kokkos::initialize( argc, argv );
  {

    if (use_batches) {
      run_batch_test(N, M, nrepeat, nbatches);
    } else {
      run_normal_test(N, M, nrepeat);
    }

  }
  Kokkos::finalize();

  return 0;
}


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



void run_batch_test(int N, int M, int nrepeat, int nbatches) {
  
  // Allocate y, x vectors and Matrix A on device.
  ViewMatrixBatch A( "A", N, M, nbatches );
  ViewMatrixBatch B( "B", M, N, nbatches );
  ViewMatrixBatch C( "C", N, N, nbatches );
  init_batches(A, B, C, N, M, nbatches);

  // Timer products.
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {

    batch_gemm(A, B, C, N, M, nbatches);

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

  double err = check_result_batch(C, N, M, nbatches);
  if (err > TOL)  {
    printf("ERROR in computation; result check failed; error = %f\n", err);
  }
  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s )\n",
          N, M, nrepeat, Gbytes * 1000, time);

}


void init_batches(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches) {

  // Initialize A matrix on host.
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < M; ++i ) {
      for ( int b = 0; b < nbatches; ++b ) {
        A(j, i, b) = AVAL;
      }
    }
  }

  // Initialize B matrix on host.
  for ( int j = 0; j <M; ++j ) {
    for ( int i = 0; i < N; ++i ) {
      for ( int b = 0; b < nbatches; ++b ) {
        B(j, i, b) = BVAL;
      }
    }
  }

  // Initialize C matrix on host.
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < N; ++i ) {
      for ( int b = 0; b < nbatches; ++b ) {
        C(j, i, b) = 1;
      }
    }
  }

}

void batch_gemm(ViewMatrixBatch A, ViewMatrixBatch B, ViewMatrixBatch C, int N, int M, int nbatches) {
  
  Kokkos::parallel_for( N, KOKKOS_LAMBDA ( int i ) {
  // for ( int i = 0; i < N; ++i ) {
    for ( int j = 0; j < N; ++j ) {

      for ( int b = 0; b < nbatches; ++b ) 
        C(i,j,b) = 0.0;

      for ( int k = 0; k < M; ++k ) {
        for ( int b = 0; b < nbatches; ++b ) {
          C(i,j,b) += A(i,k,b) * B(k,j,b);
        }
      }
      
    }
  // }
  });

}


double check_result_batch(ViewMatrixBatch C, int N, int M, int nbatches) {

  int i, j;

  double e  = 0.0;
  double ee = 0.0;
  double v  = AVAL * BVAL * M;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      for ( int b = 0; b < nbatches; ++b ) {
        e = C(i,j,b) - v;
        ee += e * e;
      }
    }
  }

  return ee;
}


/* Copyright for the lovely input reader below
// It comes from the kokkos examples
// but we have lightly modified it to work with nice small matrices
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
void read_input(int argc, char* argv[], int &N, int &M, int &S, int &nrepeat, int &batch_size, int &nbatches, bool &use_batches) {

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = atoi( argv[ ++i ] );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = atoi( argv[ ++i ] );;
      printf( "  User M is %d\n", M );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-batch_size" ) == 0 ) {
      batch_size = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-nbatches" ) == 0 ) {
      nbatches = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-use_batches" ) == 0 ) {
      use_batches = true;
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      determines number of rows (default: 1024)\n" );
      printf( "  -Columns (-M) <int>:   determines number of columns (default: 1024)\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -batch_size <int>:     number of matrices per batch (default: 100)\n" );
      printf( "  -nbatches <int>:       number of batches (default: 100)\n" );
      printf( "  -use_batches:          perform batch based test\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = 1024;

  // If N is undefined, set it.
  if ( N == -1 ) N = 1024;

  S = M*N;

  printf( "  Total size S = %d N = %d M = %d\n", S, N, M );
  printf( "  Using %d batches of %d matrices", 1, nbatches );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }

}








