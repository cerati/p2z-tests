#ifndef __PTOZ_DATA__
#define __PTOZ_DATA__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

#ifdef USE_CALI
#include <caliper/cali.h>
#endif

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>

#define nevts 100      // number of events
#define nb    600     // number of batches? 600
#define bsize 16       // batch size (tracks per batch?)

// #define nevts 100      // number of events
// #define nb    150      // number of batches?
// #define bsize 64       // batch size (tracks per batch?) 

#define ntrks nb*bsize // number of tracks per event?
#define smear 0.1      // for making more tracks for the one

#ifndef NITER
#define NITER 100
#endif

// TODO adjust everything so it is all full matrices
// TODO also to kokkos views eventually


struct VectorINT
{
  int* vals;     // will be num_mats
  int num_mats;  // number of matrices
};
struct VectorMP
{
  float** vals;  // will be rows X num_mats
  int num_mats;  // number of matrices
  int rows;      // number of rows
};
struct MatrixMP
{
  float*** vals; // will be rows X cols X num_mats
  int num_mats;  // number of matrices
  int rows;      // number of rows
  int cols;      // number of columns
};


struct MPTRK {
  float**  par;  // batch of len 6 vectors
  float***  cov; // 6x6 symmetric batch matrix
  int* q;        // bsize array of int
  int* hitidx;   // unused; array len 22 of int
  int num_mats;  // number of matrices
  int rows;      // number of rows
  int cols;      // number of columns
};

struct MPHIT {
  float** pos;   // batch of len 3 vectors
  float*** cov;  // 6x6 symmetric batch matrix
  int num_mats;  // number of matrices
  int rows;      // number of rows
  int cols;      // number of columns
};

void allocate_MPTRK(struct MPTRK &trk);
void allocate_MPHIT(struct MPHIT &hit);
void free_MPTRK(struct MPTRK &trk);
void free_MPHIT(struct MPHIT &hit);

// for the par vectors
#define X_IND 0
#define Y_IND 1
#define Z_IND 2
#define IPT_IND 3
#define PHI_IND 4
#define THETA_IND 5

struct IntMKL
{
  int** vals; // bats X mats
};
struct VectorMKL
{
  float** vals; //will be rows X num_mats X bats
  int num_mats;  // number of matrices
  int rows;      // number of rows
};
struct MatrixMKL
{
  float** vals; //will be rows X cols X num_mats
  int num_mats;  // number of matrices per batch
  int num_bats;  // number of batches
};


struct MKLTRK {
  float**   par;    // batch of len 6 vectors
  float**  cov;    // 6x6 symmetric batch matrix
  int*       q;      // bsize array of int
  // ViewIntCB     hitidx; // unused; array len 22 of int
  int num_mats;  // number of matrices per batch
  int num_bats;  // number of batches
};

struct MKLHIT {
  float** pos;     // batch of len 3 vectors
  float** cov;     // 6x6 symmetric batch matrix
  int num_mats;  // number of matrices per batch
  int num_bats;  // number of batches
};

void allocate_MatrixMKL(struct MatrixMKL &mat, int num_bats);
void allocate_MKLTRK(struct MKLTRK &trk, int num_bats);
void allocate_MKLHIT(struct MKLHIT &hit, int num_bats);
void free_MatrixMKL(struct MatrixMKL &mat, int num_bats);
void free_MKLTRK(struct MKLTRK &trk, int num_bats);
void free_MKLHIT(struct MKLHIT &hit, int num_bats);

void mkl_compact(float **A, float **B, float **C,
                 float *_A, float *_B, float *_C,  
                 int num_mats);

// not kokkos types because it is used for initialization only
struct ATRK {
  float par[6];  // vector
  float cov[36]; // symmetric mat
  int q;
  int hitidx[22];
};
struct AHIT {
  float pos[3];
  float cov[6];
};
size_t PosInMtrx(size_t i, size_t j, size_t D);
size_t SymOffsets33(size_t i);
size_t SymOffsets66(size_t i);


#endif
