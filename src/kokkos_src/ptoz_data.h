#ifndef __PTOZ_DATA__
#define __PTOZ_DATA__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// kokkos
#include <Kokkos_Core.hpp>
// our typedefs curtisy of the examples
#include "kokkos_config.h"

#define nlayers 20      // number of events
#define nevts 100      // number of events
#define nb    ntrks/bsize     // number of batches? 600
#define bsize 16       // batch size (tracks per batch?)

// #define nevts 100      // number of events
// #define nb    150      // number of batches?
// #define bsize 64       // batch size (tracks per batch?) 

#define ntrks 9600 // number of tracks per event?
#define smear 0.1      // for making more tracks for the one


#ifndef nlayers
#define nlayers 20     // number of layers
#endif

#ifndef NITER
#define NITER 5
#endif

// TODO adjust everything so it is all full matrices
// TODO also to kokkos views eventually

// Allocate y, x vectors and Matrix A on device.
typedef Kokkos::View<float*, Layout, Kokkos::HostSpace>   ViewVector; 
typedef Kokkos::View<float**, Layout, Kokkos::HostSpace>  ViewMatrix;

typedef Kokkos::View<int*, Layout,  Kokkos::HostSpace> ViewVectorINT;
typedef Kokkos::View<float**, Layout,  Kokkos::HostSpace> ViewVectorMP;
typedef Kokkos::View<float***, Layout,  Kokkos::HostSpace> ViewMatrixMP;


struct MP1I {
  int data[1*bsize];
};

struct MP22I {
  int data[22*bsize];
};

struct MPTRK {
  ViewVectorMP  par;    // batch of len 6 vectors
  ViewMatrixMP  cov;    // 6x6 symmetric batch matrix
  ViewVectorINT q;      // bsize array of int
  //  ViewVectorINT hitidx; // unused; array len 22 of int
};

struct MPHIT {
  ViewVectorMP pos;     // batch of len 3 vectors
  ViewMatrixMP cov;     // 3x3 symmetric batch matrix
};

// for the par vectors
#define X_IND 0
#define Y_IND 1
#define Z_IND 2
#define IPT_IND 3
#define PHI_IND 4
#define THETA_IND 5


// typedef Kokkos::CudaUVMSpace MemSpace_CB;
// typedef Kokkos::OpenMP MemSpace_CB;
typedef Kokkos::View<int**,     Layout,  MemSpace_CB> ViewIntCB;    // collated batch of ints
typedef Kokkos::View<float***,  Layout,  MemSpace_CB> ViewVectorCB; // collated batch of vecs
typedef Kokkos::View<float****, Layout,  MemSpace_CB> ViewMatrixCB; // collated batch of mats
typedef Kokkos::View<float***,  Layout,  MemSpace_CB> ViewMatrixOB; // One batch of mats

struct CBTRK {
  ViewVectorCB  par;    // batch of len 6 vectors
  ViewMatrixCB  cov;    // 6x6 symmetric batch matrix
  ViewIntCB     q;      // bsize array of int
  // ViewIntCB     hitidx; // unused; array len 22 of int
};

struct CBHIT {
  ViewVectorCB pos;     // batch of len 3 vectors
  ViewMatrixCB cov;     // 3x3 symmetric batch matrix
};


/*
MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

float q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}

// are hits the new data?
const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
//
float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x(const MP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
float y(const MP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
float z(const MP3F* hpos, size_t it)    { return pos(hpos, it, 2); }
//
float pos(const MPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT* hits, size_t it)    { return pos(hits, it, 0); }
float y(const MPHIT* hits, size_t it)    { return pos(hits, it, 1); }
float z(const MPHIT* hits, size_t it)    { return pos(hits, it, 2); }
//
float pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
float x(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
float y(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
float z(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }
*/



// not kokkos types because it is used for initialization only
struct ATRK {
  float par[6];  // vector
  float cov[36]; // symmetric mat
  int q;
  //  int hitidx[22];
};
struct AHIT {
  float pos[3];
  float cov[6];
};
size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}
size_t SymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}
size_t SymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}


#endif
