/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <chrono>
#include <iomanip>

//#define DUMP_OUTPUT
#define FIXED_RSEED
//#define USE_ASYNC
#ifndef USE_ASYNC
#define num_streams 1
#endif

#ifndef nevts
#define nevts 100
#endif
#ifndef bsize
#define bsize 32
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    (ntrks/bsize)
#define smear 0.00001

#ifndef NITER
#define NITER 5
#endif
#ifndef nlayer
#define nlayer 20
#endif
#ifndef num_streams
#define num_streams 10
#endif

#ifndef threadsperblockx
#define threadsperblockx bsize
#endif
#define threadsperblocky 1
#ifndef blockspergrid
#define blockspergrid nevts*nb/num_streams
#endif

#define HOSTDEV __host__ __device__

HOSTDEV size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

HOSTDEV size_t SymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

HOSTDEV size_t SymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}


struct ATRK {
  float par[6];
  float cov[21];
  int q;
//  int hitidx[22];
};

struct AHIT {
  float pos[3];
  float cov[6];
};

struct MP1I {
  int data[1*bsize];
};

struct MP22I {
  int data[22*bsize];
};

struct MP3F {
  float data[3*bsize];
};

struct MP6F {
  float data[6*bsize];
};

struct MP3x3 {
  float data[9*bsize];
};

struct MP3x6 {
  float data[18*bsize];
};

struct MP3x3SF {
  float data[6*bsize];
};

struct MP6x6SF {
  float data[21*bsize];
};

struct MP6x6F {
  float data[36*bsize];
};

struct MP2x2SF {
  float data[3*bsize];
};

struct MP2x6 {
  float data[12*bsize];
};

struct MP2F {
  float data[2*bsize];
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;
//  MP22I   hitidx;
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
};

float randn(float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;
  if (call == 1) {
    call = !call;
    return (mu + sigma * (float) X2);
  } do {
    U1 = -1 + ((float) rand () / RAND_MAX) * 2;
    U2 = -1 + ((float) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  }
  while (W >= 1 || W == 0);
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
  call = !call;
  return (mu + sigma * (float) X1);
}

MPTRK* prepareTracks(ATRK inputtrk) {
  MPTRK* result;
  cudaMallocManaged((void**)&result,nevts*nb*sizeof(MPTRK));
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //par
        for (size_t ip=0;ip<6;++ip) {
          result[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
        }
        //cov, scale by factor 100
        for (size_t ip=0;ip<21;++ip) {
          result[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
        }
        //q
        result[ib + nb*ie].q.data[it] = inputtrk.q;//can't really smear this or fit will be wrong
      }
    }
  }
  return result;
}

MPHIT* prepareHits(AHIT* inputhits) {
  MPHIT* result;
  cudaMallocManaged((void**)&result,nlayer*nevts*nb*sizeof(MPHIT));
  for (size_t lay=0;lay<nlayer;++lay) {

    struct AHIT inputhit = inputhits[lay]; 

    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
          //pos
          for (size_t ip=0;ip<3;++ip) {
            result[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
          }
          //cov
          for (size_t ip=0;ip<6;++ip) {
            result[lay+nlayer*(ib + nb*ie)].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
          }
        }
      }
    }
  }
  return result;
}


HOSTDEV MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

HOSTDEV const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}


HOSTDEV int q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}

HOSTDEV float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
HOSTDEV float x    (const MP6F* bpars, size_t it){ return par(bpars, it, 0); }
HOSTDEV float y    (const MP6F* bpars, size_t it){ return par(bpars, it, 1); }
HOSTDEV float z    (const MP6F* bpars, size_t it){ return par(bpars, it, 2); }
HOSTDEV float ipt  (const MP6F* bpars, size_t it){ return par(bpars, it, 3); }
HOSTDEV float phi  (const MP6F* bpars, size_t it){ return par(bpars, it, 4); }
HOSTDEV float theta(const MP6F* bpars, size_t it){ return par(bpars, it, 5); }

HOSTDEV float par(const MPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
HOSTDEV float x    (const MPTRK* btracks, size_t it){ return par(btracks, it, 0); }
HOSTDEV float y    (const MPTRK* btracks, size_t it){ return par(btracks, it, 1); }
HOSTDEV float z    (const MPTRK* btracks, size_t it){ return par(btracks, it, 2); }
HOSTDEV float ipt  (const MPTRK* btracks, size_t it){ return par(btracks, it, 3); }
HOSTDEV float phi  (const MPTRK* btracks, size_t it){ return par(btracks, it, 4); }
HOSTDEV float theta(const MPTRK* btracks, size_t it){ return par(btracks, it, 5); }

HOSTDEV float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}

HOSTDEV float x    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
HOSTDEV float y    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
HOSTDEV float z    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
HOSTDEV float ipt  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
HOSTDEV float phi  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
HOSTDEV float theta(const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }

HOSTDEV void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
HOSTDEV void setx    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 0, val); }
HOSTDEV void sety    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 1, val); }
HOSTDEV void setz    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 2, val); }
HOSTDEV void setipt  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 3, val); }
HOSTDEV void setphi  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 4, val); }
HOSTDEV void settheta(MP6F* bpars, size_t it, float val){ setpar(bpars, it, 5, val); }

HOSTDEV void setpar(MPTRK* btracks, size_t it, size_t ipar, float val){
  setpar(&(*btracks).par,it,ipar,val);
}
HOSTDEV void setx    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 0, val); }
HOSTDEV void sety    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 1, val); }
HOSTDEV void setz    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 2, val); }
HOSTDEV void setipt  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 3, val); }
HOSTDEV void setphi  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 4, val); }
HOSTDEV void settheta(MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 5, val); }

HOSTDEV MPHIT* bHit(MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib,int lay) {
  return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}

HOSTDEV float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
HOSTDEV float x(const MP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
HOSTDEV float y(const MP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
HOSTDEV float z(const MP3F* hpos, size_t it)    { return pos(hpos, it, 2); }

HOSTDEV float pos(const MPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t it)    { return pos(hits, it, 0); }
HOSTDEV float y(const MPHIT* hits, size_t it)    { return pos(hits, it, 1); }
HOSTDEV float z(const MPHIT* hits, size_t it)    { return pos(hits, it, 2); }

HOSTDEV float pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  //[DEBUG by Seyong on Dec. 28, 2020] add 4th argument(nlayer-1) to bHit() below.
  const MPHIT* bhits = bHit(hits, ev, ib, nlayer-1);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
HOSTDEV float y(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
HOSTDEV float z(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }



#define N bsize
__forceinline__ __device__ void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
  const float* a; //ASSUME_ALIGNED(a, 64);
  const float* b; //ASSUME_ALIGNED(b, 64);
  float* c;       //ASSUME_ALIGNED(c, 64);
  a = A->data; //ASSUME_ALIGNED(a, 64);
  b = B->data; //ASSUME_ALIGNED(b, 64);
  c = C->data;       //ASSUME_ALIGNED(c, 64);
  for(int n=threadIdx.x;n<N;n+=blockDim.x)
  {
    c[ 0*N+n] = b[ 0*N+n] + a[ 2*N+n]*b[ 3*N+n] + a[ 3*N+n]*b[ 6*N+n] + a[ 4*N+n]*b[10*N+n] + a[ 5*N+n]*b[15*N+n];
    c[ 1*N+n] = b[ 1*N+n] + a[ 2*N+n]*b[ 4*N+n] + a[ 3*N+n]*b[ 7*N+n] + a[ 4*N+n]*b[11*N+n] + a[ 5*N+n]*b[16*N+n];
    c[ 2*N+n] = b[ 3*N+n] + a[ 2*N+n]*b[ 5*N+n] + a[ 3*N+n]*b[ 8*N+n] + a[ 4*N+n]*b[12*N+n] + a[ 5*N+n]*b[17*N+n];
    c[ 3*N+n] = b[ 6*N+n] + a[ 2*N+n]*b[ 8*N+n] + a[ 3*N+n]*b[ 9*N+n] + a[ 4*N+n]*b[13*N+n] + a[ 5*N+n]*b[18*N+n];
    c[ 4*N+n] = b[10*N+n] + a[ 2*N+n]*b[12*N+n] + a[ 3*N+n]*b[13*N+n] + a[ 4*N+n]*b[14*N+n] + a[ 5*N+n]*b[19*N+n];
    c[ 5*N+n] = b[15*N+n] + a[ 2*N+n]*b[17*N+n] + a[ 3*N+n]*b[18*N+n] + a[ 4*N+n]*b[19*N+n] + a[ 5*N+n]*b[20*N+n];
    c[ 6*N+n] = b[ 1*N+n] + a[ 8*N+n]*b[ 3*N+n] + a[ 9*N+n]*b[ 6*N+n] + a[10*N+n]*b[10*N+n] + a[11*N+n]*b[15*N+n];
    c[ 7*N+n] = b[ 2*N+n] + a[ 8*N+n]*b[ 4*N+n] + a[ 9*N+n]*b[ 7*N+n] + a[10*N+n]*b[11*N+n] + a[11*N+n]*b[16*N+n];
    c[ 8*N+n] = b[ 4*N+n] + a[ 8*N+n]*b[ 5*N+n] + a[ 9*N+n]*b[ 8*N+n] + a[10*N+n]*b[12*N+n] + a[11*N+n]*b[17*N+n];
    c[ 9*N+n] = b[ 7*N+n] + a[ 8*N+n]*b[ 8*N+n] + a[ 9*N+n]*b[ 9*N+n] + a[10*N+n]*b[13*N+n] + a[11*N+n]*b[18*N+n];
    c[10*N+n] = b[11*N+n] + a[ 8*N+n]*b[12*N+n] + a[ 9*N+n]*b[13*N+n] + a[10*N+n]*b[14*N+n] + a[11*N+n]*b[19*N+n];
    c[11*N+n] = b[16*N+n] + a[ 8*N+n]*b[17*N+n] + a[ 9*N+n]*b[18*N+n] + a[10*N+n]*b[19*N+n] + a[11*N+n]*b[20*N+n];
    c[12*N+n] = 0;
    c[13*N+n] = 0;
    c[14*N+n] = 0;
    c[15*N+n] = 0;
    c[16*N+n] = 0;
    c[17*N+n] = 0;
    c[18*N+n] = b[ 6*N+n];
    c[19*N+n] = b[ 7*N+n];
    c[20*N+n] = b[ 8*N+n];
    c[21*N+n] = b[ 9*N+n];
    c[22*N+n] = b[13*N+n];
    c[23*N+n] = b[18*N+n];
    c[24*N+n] = a[26*N+n]*b[ 3*N+n] + a[27*N+n]*b[ 6*N+n] + b[10*N+n] + a[29*N+n]*b[15*N+n];
    c[25*N+n] = a[26*N+n]*b[ 4*N+n] + a[27*N+n]*b[ 7*N+n] + b[11*N+n] + a[29*N+n]*b[16*N+n];
    c[26*N+n] = a[26*N+n]*b[ 5*N+n] + a[27*N+n]*b[ 8*N+n] + b[12*N+n] + a[29*N+n]*b[17*N+n];
    c[27*N+n] = a[26*N+n]*b[ 8*N+n] + a[27*N+n]*b[ 9*N+n] + b[13*N+n] + a[29*N+n]*b[18*N+n];
    c[28*N+n] = a[26*N+n]*b[12*N+n] + a[27*N+n]*b[13*N+n] + b[14*N+n] + a[29*N+n]*b[19*N+n];
    c[29*N+n] = a[26*N+n]*b[17*N+n] + a[27*N+n]*b[18*N+n] + b[19*N+n] + a[29*N+n]*b[20*N+n];
    c[30*N+n] = b[15*N+n];
    c[31*N+n] = b[16*N+n];
    c[32*N+n] = b[17*N+n];
    c[33*N+n] = b[18*N+n];
    c[34*N+n] = b[19*N+n];
    c[35*N+n] = b[20*N+n];
  }
}

__forceinline__ __device__ void MultHelixPropTranspEndcap(MP6x6F* A, MP6x6F* B, MP6x6SF* C) {
  const float* a; //ASSUME_ALIGNED(a, 64);
  const float* b; //ASSUME_ALIGNED(b, 64);
  float* c;       //ASSUME_ALIGNED(c, 64);
  a = A->data; //ASSUME_ALIGNED(a, 64);
  b = B->data; //ASSUME_ALIGNED(b, 64);
  c = C->data;       //ASSUME_ALIGNED(c, 64);
  for(int n=threadIdx.x;n<N;n+=blockDim.x)
  {
    c[ 0*N+n] = b[ 0*N+n] + b[ 2*N+n]*a[ 2*N+n] + b[ 3*N+n]*a[ 3*N+n] + b[ 4*N+n]*a[ 4*N+n] + b[ 5*N+n]*a[ 5*N+n];
    c[ 1*N+n] = b[ 6*N+n] + b[ 8*N+n]*a[ 2*N+n] + b[ 9*N+n]*a[ 3*N+n] + b[10*N+n]*a[ 4*N+n] + b[11*N+n]*a[ 5*N+n];
    c[ 2*N+n] = b[ 7*N+n] + b[ 8*N+n]*a[ 8*N+n] + b[ 9*N+n]*a[ 9*N+n] + b[10*N+n]*a[10*N+n] + b[11*N+n]*a[11*N+n];
    c[ 3*N+n] = b[12*N+n] + b[14*N+n]*a[ 2*N+n] + b[15*N+n]*a[ 3*N+n] + b[16*N+n]*a[ 4*N+n] + b[17*N+n]*a[ 5*N+n];
    c[ 4*N+n] = b[13*N+n] + b[14*N+n]*a[ 8*N+n] + b[15*N+n]*a[ 9*N+n] + b[16*N+n]*a[10*N+n] + b[17*N+n]*a[11*N+n];
    c[ 5*N+n] = 0;
    c[ 6*N+n] = b[18*N+n] + b[20*N+n]*a[ 2*N+n] + b[21*N+n]*a[ 3*N+n] + b[22*N+n]*a[ 4*N+n] + b[23*N+n]*a[ 5*N+n];
    c[ 7*N+n] = b[19*N+n] + b[20*N+n]*a[ 8*N+n] + b[21*N+n]*a[ 9*N+n] + b[22*N+n]*a[10*N+n] + b[23*N+n]*a[11*N+n];
    c[ 8*N+n] = 0;
    c[ 9*N+n] = b[21*N+n];
    c[10*N+n] = b[24*N+n] + b[26*N+n]*a[ 2*N+n] + b[27*N+n]*a[ 3*N+n] + b[28*N+n]*a[ 4*N+n] + b[29*N+n]*a[ 5*N+n];
    c[11*N+n] = b[25*N+n] + b[26*N+n]*a[ 8*N+n] + b[27*N+n]*a[ 9*N+n] + b[28*N+n]*a[10*N+n] + b[29*N+n]*a[11*N+n];
    c[12*N+n] = 0;
    c[13*N+n] = b[27*N+n];
    c[14*N+n] = b[26*N+n]*a[26*N+n] + b[27*N+n]*a[27*N+n] + b[28*N+n] + b[29*N+n]*a[29*N+n];
    c[15*N+n] = b[30*N+n] + b[32*N+n]*a[ 2*N+n] + b[33*N+n]*a[ 3*N+n] + b[34*N+n]*a[ 4*N+n] + b[35*N+n]*a[ 5*N+n];
    c[16*N+n] = b[31*N+n] + b[32*N+n]*a[ 8*N+n] + b[33*N+n]*a[ 9*N+n] + b[34*N+n]*a[10*N+n] + b[35*N+n]*a[11*N+n];
    c[17*N+n] = 0;
    c[18*N+n] = b[33*N+n];
    c[19*N+n] = b[32*N+n]*a[26*N+n] + b[33*N+n]*a[27*N+n] + b[34*N+n] + b[35*N+n]*a[29*N+n];
    c[20*N+n] = b[35*N+n];
  }
}

__forceinline__ __device__ void KalmanGainInv(const MP6x6SF* A, const MP3x3SF* B, MP3x3* C) {
  // k = P Ht(HPHt + R)^-1
  // HpHt -> cov of x,y,z. take upper 3x3 matrix of P
  // This calculates the inverse of HpHt +R
  const float* a; //ASSUME_ALIGNED(a, 64);
  const float* b; //ASSUME_ALIGNED(b, 64);
  float* c;       //ASSUME_ALIGNED(c, 64);
  a = (*A).data; //ASSUME_ALIGNED(a, 64);
  b = (*B).data; //ASSUME_ALIGNED(b, 64);
  c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  for(int n=threadIdx.x;n<N;n+=blockDim.x)
  {
    double det =
      ((a[0*N+n]+b[0*N+n])*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])))) -
      ((a[1*N+n]+b[1*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])))) +
      ((a[2*N+n]+b[2*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n]))));
    double invdet = 1.0/det;

    c[ 0*N+n] =  invdet*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 1*N+n] =  -invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 2*N+n] =  invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 3*N+n] =  -invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 4*N+n] =  invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 5*N+n] =  -invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 6*N+n] =  invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n])));
    c[ 7*N+n] =  -invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 8*N+n] =  invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[6*N+n]+b[3*N+n])) - ((a[1*N+n]+b[1*N+n]) *(a[1*N+n]+b[1*N+n])));
  }
}

__forceinline__ __device__ void KalmanGain(const MP6x6SF* A, const MP3x3* B, MP3x6* C) {
  // k = P Ht(HPHt + R)^-1
  // HpHt -> cov of x,y,z. take upper 3x3 matrix of P
  // This calculates the kalman gain 
  const float* a; //ASSUME_ALIGNED(a, 64);
  const float* b; //ASSUME_ALIGNED(b, 64);
  float* c;       //ASSUME_ALIGNED(c, 64);
  a = (*A).data; //ASSUME_ALIGNED(a, 64);
  b = (*B).data; //ASSUME_ALIGNED(b, 64);
  c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  for(int n=threadIdx.x;n<N;n+=blockDim.x)
  {
    c[ 0*N+n] = a[0*N+n]*b[0*N+n] + a[1*N+n]*b[3*N+n] + a[2*N+n]*b[6*N+n];
    c[ 1*N+n] = a[0*N+n]*b[1*N+n] + a[1*N+n]*b[4*N+n] + a[2*N+n]*b[7*N+n];
    c[ 2*N+n] = a[0*N+n]*b[2*N+n] + a[1*N+n]*b[5*N+n] + a[2*N+n]*b[8*N+n];
    c[ 3*N+n] = a[1*N+n]*b[0*N+n] + a[6*N+n]*b[3*N+n] + a[7*N+n]*b[6*N+n];
    c[ 4*N+n] = a[1*N+n]*b[1*N+n] + a[6*N+n]*b[4*N+n] + a[7*N+n]*b[7*N+n];
    c[ 5*N+n] = a[1*N+n]*b[2*N+n] + a[6*N+n]*b[5*N+n] + a[7*N+n]*b[8*N+n];
    c[ 6*N+n] = a[2*N+n]*b[0*N+n] + a[7*N+n]*b[3*N+n] + a[11*N+n]*b[6*N+n];
    c[ 7*N+n] = a[2*N+n]*b[1*N+n] + a[7*N+n]*b[4*N+n] + a[11*N+n]*b[7*N+n];
    c[ 8*N+n] = a[2*N+n]*b[2*N+n] + a[7*N+n]*b[5*N+n] + a[11*N+n]*b[8*N+n];
    c[ 9*N+n] = a[3*N+n]*b[0*N+n] + a[8*N+n]*b[3*N+n] + a[12*N+n]*b[6*N+n];
    c[ 10*N+n] = a[3*N+n]*b[1*N+n] + a[8*N+n]*b[4*N+n] + a[12*N+n]*b[7*N+n];
    c[ 11*N+n] = a[3*N+n]*b[2*N+n] + a[8*N+n]*b[5*N+n] + a[12*N+n]*b[8*N+n];
    c[ 12*N+n] = a[4*N+n]*b[0*N+n] + a[9*N+n]*b[3*N+n] + a[13*N+n]*b[6*N+n];
    c[ 13*N+n] = a[4*N+n]*b[1*N+n] + a[9*N+n]*b[4*N+n] + a[13*N+n]*b[7*N+n];
    c[ 14*N+n] = a[4*N+n]*b[2*N+n] + a[9*N+n]*b[5*N+n] + a[13*N+n]*b[8*N+n];
    c[ 15*N+n] = a[5*N+n]*b[0*N+n] + a[10*N+n]*b[3*N+n] + a[14*N+n]*b[6*N+n];
    c[ 16*N+n] = a[5*N+n]*b[1*N+n] + a[10*N+n]*b[4*N+n] + a[14*N+n]*b[7*N+n];
    c[ 17*N+n] = a[5*N+n]*b[2*N+n] + a[10*N+n]*b[5*N+n] + a[14*N+n]*b[8*N+n];
  }
}

__forceinline__ __device__ void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){
  __shared__ MP3x3 inverse_temp;
  __shared__ MP3x6 kGain;
  __shared__ MP6x6SF newErr;
  KalmanGainInv(trkErr,hitErr,&inverse_temp);
  KalmanGain(trkErr,&inverse_temp,&kGain);

  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x){
    const float xin = x(inPar,it);
    const float yin = y(inPar,it);
    const float zin = z(inPar,it);
    const float ptin = 1.0f/ipt(inPar,it); // is this pt or ipt? 
    const float phiin = phi(inPar,it);
    const float thetain = theta(inPar,it);
    const float xout = x(msP,it);
    const float yout = y(msP,it);
    //const float zout = z(msP,it);
  
    float xnew = xin + (kGain.data[0*bsize+it]*(xout-xin)) +(kGain.data[1*bsize+it]*(yout-yin));
    float ynew = yin + (kGain.data[3*bsize+it]*(xout-xin)) +(kGain.data[4*bsize+it]*(yout-yin));
    float znew = zin + (kGain.data[6*bsize+it]*(xout-xin)) +(kGain.data[7*bsize+it]*(yout-yin));
    float ptnew = ptin + (kGain.data[9*bsize+it]*(xout-xin)) +(kGain.data[10*bsize+it]*(yout-yin));
    float phinew = phiin + (kGain.data[12*bsize+it]*(xout-xin)) +(kGain.data[13*bsize+it]*(yout-yin));
    float thetanew = thetain + (kGain.data[15*bsize+it]*(xout-xin)) +(kGain.data[16*bsize+it]*(yout-yin));
  
    newErr.data[0*bsize+it] = trkErr->data[0*bsize+it] - (kGain.data[0*bsize+it]*trkErr->data[0*bsize+it]+kGain.data[1*bsize+it]*trkErr->data[1*bsize+it]+kGain.data[2*bsize+it]*trkErr->data[2*bsize+it]);
    newErr.data[1*bsize+it] = trkErr->data[1*bsize+it] - (kGain.data[0*bsize+it]*trkErr->data[1*bsize+it]+kGain.data[1*bsize+it]*trkErr->data[6*bsize+it]+kGain.data[2*bsize+it]*trkErr->data[7*bsize+it]);
    newErr.data[2*bsize+it] = trkErr->data[2*bsize+it] - (kGain.data[0*bsize+it]*trkErr->data[2*bsize+it]+kGain.data[1*bsize+it]*trkErr->data[7*bsize+it]+kGain.data[2*bsize+it]*trkErr->data[11*bsize+it]);
    newErr.data[3*bsize+it] = trkErr->data[3*bsize+it] - (kGain.data[0*bsize+it]*trkErr->data[3*bsize+it]+kGain.data[1*bsize+it]*trkErr->data[8*bsize+it]+kGain.data[2*bsize+it]*trkErr->data[12*bsize+it]);
    newErr.data[4*bsize+it] = trkErr->data[4*bsize+it] - (kGain.data[0*bsize+it]*trkErr->data[4*bsize+it]+kGain.data[1*bsize+it]*trkErr->data[9*bsize+it]+kGain.data[2*bsize+it]*trkErr->data[13*bsize+it]);
    newErr.data[5*bsize+it] = trkErr->data[5*bsize+it] - (kGain.data[0*bsize+it]*trkErr->data[5*bsize+it]+kGain.data[1*bsize+it]*trkErr->data[10*bsize+it]+kGain.data[2*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr.data[6*bsize+it] = trkErr->data[6*bsize+it] - (kGain.data[3*bsize+it]*trkErr->data[1*bsize+it]+kGain.data[4*bsize+it]*trkErr->data[6*bsize+it]+kGain.data[5*bsize+it]*trkErr->data[7*bsize+it]);
    newErr.data[7*bsize+it] = trkErr->data[7*bsize+it] - (kGain.data[3*bsize+it]*trkErr->data[2*bsize+it]+kGain.data[4*bsize+it]*trkErr->data[7*bsize+it]+kGain.data[5*bsize+it]*trkErr->data[11*bsize+it]);
    newErr.data[8*bsize+it] = trkErr->data[8*bsize+it] - (kGain.data[3*bsize+it]*trkErr->data[3*bsize+it]+kGain.data[4*bsize+it]*trkErr->data[8*bsize+it]+kGain.data[5*bsize+it]*trkErr->data[12*bsize+it]);
    newErr.data[9*bsize+it] = trkErr->data[9*bsize+it] - (kGain.data[3*bsize+it]*trkErr->data[4*bsize+it]+kGain.data[4*bsize+it]*trkErr->data[9*bsize+it]+kGain.data[5*bsize+it]*trkErr->data[13*bsize+it]);
    newErr.data[10*bsize+it] = trkErr->data[10*bsize+it] - (kGain.data[3*bsize+it]*trkErr->data[5*bsize+it]+kGain.data[4*bsize+it]*trkErr->data[10*bsize+it]+kGain.data[5*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr.data[11*bsize+it] = trkErr->data[11*bsize+it] - (kGain.data[6*bsize+it]*trkErr->data[2*bsize+it]+kGain.data[7*bsize+it]*trkErr->data[7*bsize+it]+kGain.data[8*bsize+it]*trkErr->data[11*bsize+it]);
    newErr.data[12*bsize+it] = trkErr->data[12*bsize+it] - (kGain.data[6*bsize+it]*trkErr->data[3*bsize+it]+kGain.data[7*bsize+it]*trkErr->data[8*bsize+it]+kGain.data[8*bsize+it]*trkErr->data[12*bsize+it]);
    newErr.data[13*bsize+it] = trkErr->data[13*bsize+it] - (kGain.data[6*bsize+it]*trkErr->data[4*bsize+it]+kGain.data[7*bsize+it]*trkErr->data[9*bsize+it]+kGain.data[8*bsize+it]*trkErr->data[13*bsize+it]);
    newErr.data[14*bsize+it] = trkErr->data[14*bsize+it] - (kGain.data[6*bsize+it]*trkErr->data[5*bsize+it]+kGain.data[7*bsize+it]*trkErr->data[10*bsize+it]+kGain.data[8*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr.data[15*bsize+it] = trkErr->data[15*bsize+it] - (kGain.data[9*bsize+it]*trkErr->data[3*bsize+it]+kGain.data[10*bsize+it]*trkErr->data[8*bsize+it]+kGain.data[11*bsize+it]*trkErr->data[12*bsize+it]);
    newErr.data[16*bsize+it] = trkErr->data[16*bsize+it] - (kGain.data[9*bsize+it]*trkErr->data[4*bsize+it]+kGain.data[10*bsize+it]*trkErr->data[9*bsize+it]+kGain.data[11*bsize+it]*trkErr->data[13*bsize+it]);
    newErr.data[17*bsize+it] = trkErr->data[17*bsize+it] - (kGain.data[9*bsize+it]*trkErr->data[5*bsize+it]+kGain.data[10*bsize+it]*trkErr->data[10*bsize+it]+kGain.data[11*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr.data[18*bsize+it] = trkErr->data[18*bsize+it] - (kGain.data[12*bsize+it]*trkErr->data[4*bsize+it]+kGain.data[13*bsize+it]*trkErr->data[9*bsize+it]+kGain.data[14*bsize+it]*trkErr->data[13*bsize+it]);
    newErr.data[19*bsize+it] = trkErr->data[19*bsize+it] - (kGain.data[12*bsize+it]*trkErr->data[5*bsize+it]+kGain.data[13*bsize+it]*trkErr->data[10*bsize+it]+kGain.data[14*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr.data[20*bsize+it] = trkErr->data[20*bsize+it] - (kGain.data[15*bsize+it]*trkErr->data[5*bsize+it]+kGain.data[16*bsize+it]*trkErr->data[10*bsize+it]+kGain.data[17*bsize+it]*trkErr->data[14*bsize+it]);
  
    setx(inPar,it,xnew );
    sety(inPar,it,ynew );
    setz(inPar,it,znew);
    setipt(inPar,it, ptnew);
    setphi(inPar,it, phinew);
    settheta(inPar,it, thetanew);
  }
  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x){
    #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr->data[ i*bsize+it] = trkErr->data[ i*bsize+it] - newErr.data[ i*bsize+it];
    }
  }
}

__forceinline__ __device__ void KalmanUpdate_v2(struct MP6x6SF* trkErr, struct MP6F* inPar, const struct MP3x3SF* hitErr, const struct MP3F* msP, struct MP2x2SF* resErr_loc, struct MP2x6* kGain, struct MP2F* res_loc, struct MP6x6SF* newErr){

   // AddIntoUpperLeft2x2(psErr, msErr, resErr);
   for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x)
   {
     resErr_loc->data[0*bsize+it] = trkErr->data[0*bsize+it] + hitErr->data[0*bsize+it];
     resErr_loc->data[1*bsize+it] = trkErr->data[1*bsize+it] + hitErr->data[1*bsize+it];
     resErr_loc->data[2*bsize+it] = trkErr->data[2*bsize+it] + hitErr->data[2*bsize+it];
   }

   // Matriplex::InvertCramerSym(resErr);
   for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x)
   {
     const double det = (double)resErr_loc->data[0*bsize+it] * resErr_loc->data[2*bsize+it] -
                        (double)resErr_loc->data[1*bsize+it] * resErr_loc->data[1*bsize+it];
     const float s   = 1.f / det;
     const float tmp = s * resErr_loc->data[2*bsize+it];
     resErr_loc->data[1*bsize+it] *= -s;
     resErr_loc->data[2*bsize+it]  = s * resErr_loc->data[0*bsize+it];
     resErr_loc->data[0*bsize+it]  = tmp;
   }

   // KalmanGain(psErr, resErr, K);
   for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x)
   {
      kGain->data[ 0*bsize+it] = trkErr->data[ 0*bsize+it]*resErr_loc->data[ 0*bsize+it] + trkErr->data[ 1*bsize+it]*resErr_loc->data[ 1*bsize+it];
      kGain->data[ 1*bsize+it] = trkErr->data[ 0*bsize+it]*resErr_loc->data[ 1*bsize+it] + trkErr->data[ 1*bsize+it]*resErr_loc->data[ 2*bsize+it];
      kGain->data[ 2*bsize+it] = trkErr->data[ 1*bsize+it]*resErr_loc->data[ 0*bsize+it] + trkErr->data[ 2*bsize+it]*resErr_loc->data[ 1*bsize+it];
      kGain->data[ 3*bsize+it] = trkErr->data[ 1*bsize+it]*resErr_loc->data[ 1*bsize+it] + trkErr->data[ 2*bsize+it]*resErr_loc->data[ 2*bsize+it];
      kGain->data[ 4*bsize+it] = trkErr->data[ 3*bsize+it]*resErr_loc->data[ 0*bsize+it] + trkErr->data[ 4*bsize+it]*resErr_loc->data[ 1*bsize+it];
      kGain->data[ 5*bsize+it] = trkErr->data[ 3*bsize+it]*resErr_loc->data[ 1*bsize+it] + trkErr->data[ 4*bsize+it]*resErr_loc->data[ 2*bsize+it];
      kGain->data[ 6*bsize+it] = trkErr->data[ 6*bsize+it]*resErr_loc->data[ 0*bsize+it] + trkErr->data[ 7*bsize+it]*resErr_loc->data[ 1*bsize+it];
      kGain->data[ 7*bsize+it] = trkErr->data[ 6*bsize+it]*resErr_loc->data[ 1*bsize+it] + trkErr->data[ 7*bsize+it]*resErr_loc->data[ 2*bsize+it];
      kGain->data[ 8*bsize+it] = trkErr->data[10*bsize+it]*resErr_loc->data[ 0*bsize+it] + trkErr->data[11*bsize+it]*resErr_loc->data[ 1*bsize+it];
      kGain->data[ 9*bsize+it] = trkErr->data[10*bsize+it]*resErr_loc->data[ 1*bsize+it] + trkErr->data[11*bsize+it]*resErr_loc->data[ 2*bsize+it];
      kGain->data[10*bsize+it] = trkErr->data[15*bsize+it]*resErr_loc->data[ 0*bsize+it] + trkErr->data[16*bsize+it]*resErr_loc->data[ 1*bsize+it];
      kGain->data[11*bsize+it] = trkErr->data[15*bsize+it]*resErr_loc->data[ 1*bsize+it] + trkErr->data[16*bsize+it]*resErr_loc->data[ 2*bsize+it];
   }

   // SubtractFirst2(msPar, psPar, res);
   // MultResidualsAdd(K, psPar, res, outPar);
   for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x)
   {
     res_loc->data[0*bsize+it] =  x(msP,it) - x(inPar,it);
     res_loc->data[1*bsize+it] =  y(msP,it) - y(inPar,it);

     setx    (inPar, it, x    (inPar, it) + kGain->data[ 0*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 1*bsize+it] * res_loc->data[ 1*bsize+it]);
     sety    (inPar, it, y    (inPar, it) + kGain->data[ 2*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 3*bsize+it] * res_loc->data[ 1*bsize+it]);
     setz    (inPar, it, z    (inPar, it) + kGain->data[ 4*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 5*bsize+it] * res_loc->data[ 1*bsize+it]);
     setipt  (inPar, it, ipt  (inPar, it) + kGain->data[ 6*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 7*bsize+it] * res_loc->data[ 1*bsize+it]);
     setphi  (inPar, it, phi  (inPar, it) + kGain->data[ 8*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 9*bsize+it] * res_loc->data[ 1*bsize+it]);
     settheta(inPar, it, theta(inPar, it) + kGain->data[10*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[11*bsize+it] * res_loc->data[ 1*bsize+it]);
     //note: if ipt changes sign we should update the charge, or we should get rid of the charge altogether and just use the sign of ipt
   }

   // squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
   // missing

   // KHC(K, psErr, outErr);
   // outErr.Subtract(psErr, outErr);
   for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x)
   {
      newErr->data[ 0*bsize+it] = kGain->data[ 0*bsize+it]*trkErr->data[ 0*bsize+it] + kGain->data[ 1*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr->data[ 1*bsize+it] = kGain->data[ 2*bsize+it]*trkErr->data[ 0*bsize+it] + kGain->data[ 3*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr->data[ 2*bsize+it] = kGain->data[ 2*bsize+it]*trkErr->data[ 1*bsize+it] + kGain->data[ 3*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr->data[ 3*bsize+it] = kGain->data[ 4*bsize+it]*trkErr->data[ 0*bsize+it] + kGain->data[ 5*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr->data[ 4*bsize+it] = kGain->data[ 4*bsize+it]*trkErr->data[ 1*bsize+it] + kGain->data[ 5*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr->data[ 5*bsize+it] = kGain->data[ 4*bsize+it]*trkErr->data[ 3*bsize+it] + kGain->data[ 5*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr->data[ 6*bsize+it] = kGain->data[ 6*bsize+it]*trkErr->data[ 0*bsize+it] + kGain->data[ 7*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr->data[ 7*bsize+it] = kGain->data[ 6*bsize+it]*trkErr->data[ 1*bsize+it] + kGain->data[ 7*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr->data[ 8*bsize+it] = kGain->data[ 6*bsize+it]*trkErr->data[ 3*bsize+it] + kGain->data[ 7*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr->data[ 9*bsize+it] = kGain->data[ 6*bsize+it]*trkErr->data[ 6*bsize+it] + kGain->data[ 7*bsize+it]*trkErr->data[ 7*bsize+it];
      newErr->data[10*bsize+it] = kGain->data[ 8*bsize+it]*trkErr->data[ 0*bsize+it] + kGain->data[ 9*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr->data[11*bsize+it] = kGain->data[ 8*bsize+it]*trkErr->data[ 1*bsize+it] + kGain->data[ 9*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr->data[12*bsize+it] = kGain->data[ 8*bsize+it]*trkErr->data[ 3*bsize+it] + kGain->data[ 9*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr->data[13*bsize+it] = kGain->data[ 8*bsize+it]*trkErr->data[ 6*bsize+it] + kGain->data[ 9*bsize+it]*trkErr->data[ 7*bsize+it];
      newErr->data[14*bsize+it] = kGain->data[ 8*bsize+it]*trkErr->data[10*bsize+it] + kGain->data[ 9*bsize+it]*trkErr->data[11*bsize+it];
      newErr->data[15*bsize+it] = kGain->data[10*bsize+it]*trkErr->data[ 0*bsize+it] + kGain->data[11*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr->data[16*bsize+it] = kGain->data[10*bsize+it]*trkErr->data[ 1*bsize+it] + kGain->data[11*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr->data[17*bsize+it] = kGain->data[10*bsize+it]*trkErr->data[ 3*bsize+it] + kGain->data[11*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr->data[18*bsize+it] = kGain->data[10*bsize+it]*trkErr->data[ 6*bsize+it] + kGain->data[11*bsize+it]*trkErr->data[ 7*bsize+it];
      newErr->data[19*bsize+it] = kGain->data[10*bsize+it]*trkErr->data[10*bsize+it] + kGain->data[11*bsize+it]*trkErr->data[11*bsize+it];
      newErr->data[20*bsize+it] = kGain->data[10*bsize+it]*trkErr->data[15*bsize+it] + kGain->data[11*bsize+it]*trkErr->data[16*bsize+it];

      newErr->data[ 0*bsize+it] = trkErr->data[ 0*bsize+it] - newErr->data[ 0*bsize+it];
      newErr->data[ 1*bsize+it] = trkErr->data[ 1*bsize+it] - newErr->data[ 1*bsize+it];
      newErr->data[ 2*bsize+it] = trkErr->data[ 2*bsize+it] - newErr->data[ 2*bsize+it];
      newErr->data[ 3*bsize+it] = trkErr->data[ 3*bsize+it] - newErr->data[ 3*bsize+it];
      newErr->data[ 4*bsize+it] = trkErr->data[ 4*bsize+it] - newErr->data[ 4*bsize+it];
      newErr->data[ 5*bsize+it] = trkErr->data[ 5*bsize+it] - newErr->data[ 5*bsize+it];
      newErr->data[ 6*bsize+it] = trkErr->data[ 6*bsize+it] - newErr->data[ 6*bsize+it];
      newErr->data[ 7*bsize+it] = trkErr->data[ 7*bsize+it] - newErr->data[ 7*bsize+it];
      newErr->data[ 8*bsize+it] = trkErr->data[ 8*bsize+it] - newErr->data[ 8*bsize+it];
      newErr->data[ 9*bsize+it] = trkErr->data[ 9*bsize+it] - newErr->data[ 9*bsize+it];
      newErr->data[10*bsize+it] = trkErr->data[10*bsize+it] - newErr->data[10*bsize+it];
      newErr->data[11*bsize+it] = trkErr->data[11*bsize+it] - newErr->data[11*bsize+it];
      newErr->data[12*bsize+it] = trkErr->data[12*bsize+it] - newErr->data[12*bsize+it];
      newErr->data[13*bsize+it] = trkErr->data[13*bsize+it] - newErr->data[13*bsize+it];
      newErr->data[14*bsize+it] = trkErr->data[14*bsize+it] - newErr->data[14*bsize+it];
      newErr->data[15*bsize+it] = trkErr->data[15*bsize+it] - newErr->data[15*bsize+it];
      newErr->data[16*bsize+it] = trkErr->data[16*bsize+it] - newErr->data[16*bsize+it];
      newErr->data[17*bsize+it] = trkErr->data[17*bsize+it] - newErr->data[17*bsize+it];
      newErr->data[18*bsize+it] = trkErr->data[18*bsize+it] - newErr->data[18*bsize+it];
      newErr->data[19*bsize+it] = trkErr->data[19*bsize+it] - newErr->data[19*bsize+it];
      newErr->data[20*bsize+it] = trkErr->data[20*bsize+it] - newErr->data[20*bsize+it];
   }

  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x)
  {
    #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr->data[ i*bsize+it] = trkErr->data[ i*bsize+it] - newErr->data[ i*bsize+it];
    }
  }
}

__device__ __constant__ float kfact = 100./(-0.299792458*3.8112);
__device__ __forceinline__ void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar, const MP1I* inChg,const MP3F* msP, 
			  MP6x6SF* outErr, MP6F* outPar, struct MP6x6F* errorProp, struct MP6x6F* temp) {
        //struct MP6x6F* errorProp, temp; 
  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x){
    const float zout = z(msP,it);
    const float k = q(inChg,it)*kfact;//*100/3.8;
    const float deltaZ = zout - z(inPar,it);
    const float pt = 1.0f/ipt(inPar,it);
    const float cosP = cosf(phi(inPar,it));
    const float sinP = sinf(phi(inPar,it));
    const float cosT = cosf(theta(inPar,it));
    const float sinT = sinf(theta(inPar,it));
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0f/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*ipt(inPar,it)*icosTk;
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;
    setx(outPar,it, x(inPar,it) + k*(pxin*sina - pyin*(1.0f-cosa)) );
    sety(outPar,it, y(inPar,it) + k*(pyin*sina + pxin*(1.0f-cosa)) );
    setz(outPar,it,zout);
    setipt(outPar,it, ipt(inPar,it));
    setphi(outPar,it, phi(inPar,it)+alpha );
    settheta(outPar,it, theta(inPar,it) );
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);

    //for (size_t i=0;i<6;++i) errorProp->data[bsize*PosInMtrx(i,i,6) + it] = 1.0f;
    errorProp->data[bsize*PosInMtrx(0,0,6) + it] = 1.0f;
    errorProp->data[bsize*PosInMtrx(1,1,6) + it] = 1.0f;
    errorProp->data[bsize*PosInMtrx(2,2,6) + it] = 1.0f;
    errorProp->data[bsize*PosInMtrx(3,3,6) + it] = 1.0f;
    errorProp->data[bsize*PosInMtrx(4,4,6) + it] = 1.0f;
    errorProp->data[bsize*PosInMtrx(5,5,6) + it] = 1.0f;
    //[Dec. 21, 2022] Added to have the same pattern as the cudauvm version.
    errorProp->data[bsize*PosInMtrx(0,1,6) + it] = 0.0f;
    errorProp->data[bsize*PosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp->data[bsize*PosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.0f-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.0f-cCosPsina))*(pt*pt);
    errorProp->data[bsize*PosInMtrx(0,4,6) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.0f-cCosPsina));
    errorProp->data[bsize*PosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.0f-sinP*sCosPsina)*(icosT*icosT);
    errorProp->data[bsize*PosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp->data[bsize*PosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.0f-cCosPsina))*(pt*pt);
    errorProp->data[bsize*PosInMtrx(1,4,6) + it] = (k*pt)*(-sinP*(1.0f-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp->data[bsize*PosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp->data[bsize*PosInMtrx(4,2,6) + it] = -ipt(inPar,it)*sinT*(icosTk);
    errorProp->data[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ*(icosTk);
    errorProp->data[bsize*PosInMtrx(4,5,6) + it] = ipt(inPar,it)*deltaZ*(icosT*icosTk);
  }
  MultHelixPropEndcap(errorProp, inErr, temp);
  MultHelixPropTranspEndcap(errorProp, temp, outErr);
}



__device__ __constant__ int ie_range = (int) nevts/num_streams; 
__device__ __constant__ int ie_rangeR = (int) nevts%num_streams; 
__global__ void GPUsequence(MPTRK* trk, MPHIT* hit, MPTRK* outtrk, const int stream){
      __shared__ struct MP6x6F errorProp, temp;
      __shared__ struct MP2x2SF resErr_loc;
      __shared__ struct MP2x6 kGain;
      __shared__ struct MP2F res_loc;
      __shared__ struct MP6x6SF newErr;
      const MPTRK* __shared__ btracks;
      MPTRK* __shared__ obtracks;
      const MPHIT* __shared__ bhits;
      __shared__ int ie;
      __shared__ int ib;
      int ti;
      int lnb = nb;
  for (ti = blockIdx.x; ti<ie_range*nb; ti+=gridDim.x){
      if(threadIdx.x == 0) {
          ie = ti/lnb;
          ib = ti%lnb;
          btracks = bTk(trk,ie,ib);
          obtracks = bTk(outtrk,ie,ib);
          (*obtracks) = (*btracks);
      }
      for (int layer=0;layer<nlayer;++layer){	
        if(threadIdx.x == 0) {
            bhits = bHit(hit,ie,ib,layer);
        }
        __syncthreads();
     
        propagateToZ(&(*obtracks).cov, &(*obtracks).par, &(*obtracks).q, &(*bhits).pos, 
                     &(*obtracks).cov, &(*obtracks).par, &errorProp, &temp);
        //KalmanUpdate(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos);
        KalmanUpdate_v2(&(*obtracks).cov, &(*obtracks).par, &(*bhits).cov, &(*bhits).pos, &resErr_loc, &kGain, &res_loc, &newErr);
      }
  }
}
__global__ void GPUsequenceR(MPTRK* trk, MPHIT* hit, MPTRK* outtrk, const int stream){
      __shared__ struct MP6x6F errorProp, temp;
      __shared__ struct MP2x2SF resErr_loc;
      __shared__ struct MP2x6 kGain;
      __shared__ struct MP2F res_loc;
      __shared__ struct MP6x6SF newErr;
      const MPTRK* __shared__ btracks;
      MPTRK* __shared__ obtracks;
      const MPHIT* __shared__ bhits;
      __shared__ size_t ie;
      __shared__ size_t ib;
  for (size_t ti = blockIdx.x; ti<ie_rangeR*nb; ti+=gridDim.x){
      if(threadIdx.x == 0) {
          ie = ti/nb;
          ib = ti%nb;
          btracks = bTk(trk,ie,ib);
          obtracks = bTk(outtrk,ie,ib);
          (*obtracks) = (*btracks);
      }
      for (int layer=0;layer<nlayer;++layer){	
        if(threadIdx.x == 0) {
            bhits = bHit(hit,ie,ib,layer);
        }
        __syncthreads();

        propagateToZ(&(*obtracks).cov, &(*obtracks).par, &(*obtracks).q, &(*bhits).pos, 
                     &(*obtracks).cov, &(*obtracks).par, &errorProp, &temp);
        //KalmanUpdate(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos);
        KalmanUpdate_v2(&(*obtracks).cov, &(*obtracks).par, &(*bhits).cov, &(*bhits).pos, &resErr_loc, &kGain, &res_loc, &newErr);
      }
  }
}

void prefetch_device(MPTRK* trk, MPHIT* hit, cudaStream_t* streams, int stream_chunk, int stream_remainder, int device) {
    for (int s = 0; s<num_streams;s++){
#ifdef USE_ASYNC
      cudaMemPrefetchAsync(trk+(s*stream_chunk),stream_chunk*sizeof(MPTRK), device,streams[s]);
#else
      cudaMemPrefetchAsync(trk+(s*stream_chunk),stream_chunk*sizeof(MPTRK), device,0);
#endif
      
#ifdef USE_ASYNC
      cudaMemPrefetchAsync(hit+(s*stream_chunk*nlayer),nlayer*stream_chunk*sizeof(MPHIT), device,streams[s]);
#else
      cudaMemPrefetchAsync(hit+(s*stream_chunk*nlayer),nlayer*stream_chunk*sizeof(MPHIT), device,0);
#endif
    }  
    if(stream_remainder != 0){
#ifdef USE_ASYNC
      cudaMemPrefetchAsync(trk+(num_streams*stream_chunk),stream_remainder*sizeof(MPTRK), device,streams[num_streams]);
#else
      cudaMemPrefetchAsync(trk+(num_streams*stream_chunk),stream_remainder*sizeof(MPTRK), device,0);
#endif
      
#ifdef USE_ASYNC
      cudaMemPrefetchAsync(hit+(num_streams*stream_chunk*nlayer),nlayer*stream_remainder*sizeof(MPHIT), device,streams[num_streams]);
#else
      cudaMemPrefetchAsync(hit+(num_streams*stream_chunk*nlayer),nlayer*stream_remainder*sizeof(MPHIT), device,0);
#endif
    }
}

void prefetch_host(MPTRK* outtrk, cudaStream_t* streams, int stream_chunk, int stream_remainder) {
    for (int s = 0; s<num_streams;s++){
#ifdef USE_ASYNC
      cudaMemPrefetchAsync(outtrk+(s*stream_chunk),stream_chunk*sizeof(MPTRK), cudaCpuDeviceId,streams[s]);
#else
      cudaMemPrefetchAsync(outtrk+(s*stream_chunk),stream_chunk*sizeof(MPTRK), cudaCpuDeviceId,0);
#endif
    }
    if(stream_remainder != 0){
#ifdef USE_ASYNC
      cudaMemPrefetchAsync(outtrk+(num_streams*stream_chunk),stream_remainder*sizeof(MPTRK), cudaCpuDeviceId,streams[num_streams]);
#else
      cudaMemPrefetchAsync(outtrk+(num_streams*stream_chunk),stream_remainder*sizeof(MPTRK), cudaCpuDeviceId,0);
#endif
    }
}

int main (int argc, char* argv[]) {

#ifdef USE_ASYNC
  printf("RUNNING CUDA Async Version!!\n");
#else
  printf("RUNNING CUDA Sync Version!!\n");
#endif
#ifdef include_data
  printf("Measure Both Memory Transfer Times and Compute Times!\n");
#else
  printf("Measure Compute Times Only!\n");
#endif

#include "input_track.h"

   struct AHIT inputhits[26] = {inputhit25,inputhit24,inputhit23,inputhit22,inputhit21,inputhit20,inputhit19,inputhit18,inputhit17,
				inputhit16,inputhit15,inputhit14,inputhit13,inputhit12,inputhit11,inputhit10,inputhit09,inputhit08,
				inputhit07,inputhit06,inputhit05,inputhit04,inputhit03,inputhit02,inputhit01,inputhit00};

   printf("track in pos: x=%f, y=%f, z=%f, r=%f, pt=%f, phi=%f, theta=%f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2],
	  sqrtf(inputtrk.par[0]*inputtrk.par[0] + inputtrk.par[1]*inputtrk.par[1]),
	  1./inputtrk.par[3], inputtrk.par[4], inputtrk.par[5]);

   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66(PosInMtrx(0,0,6))],
                                               inputtrk.cov[SymOffsets66(PosInMtrx(1,1,6))],
	                                       inputtrk.cov[SymOffsets66(PosInMtrx(2,2,6))]);
   for (size_t lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }

   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);
 
  long setup_start, setup_stop;
  struct timeval timecheck;

  gettimeofday(&timecheck, NULL);
  setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;      
#ifdef FIXED_RSEED
  //[DEBUG by Seyong on Dec. 28, 2020] add an explicit srand(1) call to generate fixed inputs for better debugging.
  srand(1);
#endif
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//  cudaFuncSetCacheConfig(GPUsequence,cudaFuncCachePreferL1);
//  cudaFuncSetCacheConfig(GPUsequenceR,cudaFuncCachePreferL1);
  MPTRK* trk = prepareTracks(inputtrk);
  MPHIT* hit = prepareHits(inputhits);
  MPTRK* outtrk;
  cudaMallocManaged((void**)&outtrk,nevts*nb*sizeof(MPTRK)); 
  dim3 grid(blockspergrid,1,1);
  dim3 block(threadsperblockx,threadsperblocky,1); 
  int device = -1;
  cudaGetDevice(&device);
  int stream_chunk = ((int)(nevts/num_streams))*nb;//*sizeof(MPTRK);
  int stream_remainder = ((int)(nevts%num_streams))*nb;//*sizeof(MPTRK);
  int stream_range;
  if (stream_remainder == 0){ stream_range =num_streams;}
  else{stream_range = num_streams+1;}
  cudaStream_t streams[stream_range];
  for (int s = 0; s<stream_range;s++){
    //cudaStreamCreateWithFlags(&streams[s],cudaStreamNonBlocking);
    cudaStreamCreate(&streams[s]);
  }
#ifndef include_data
	prefetch_device(trk, hit, streams, stream_chunk, stream_remainder, device);
#ifdef USE_ASYNC
    cudaDeviceSynchronize(); 
#endif
#endif

  gettimeofday(&timecheck, NULL);
  setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

  printf("done preparing!\n");

  printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
  printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
  printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(struct MPHIT));
  
  auto wall_start = std::chrono::high_resolution_clock::now();

  int itr;
  for(itr=0; itr<NITER; itr++){
#ifdef include_data
	prefetch_device(trk, hit, streams, stream_chunk, stream_remainder, device);
#endif

    for (int s = 0; s<num_streams;++s){
      //printf("stream = %d, grid (%d, %d, %d), block(%d, %d, %d), stream_chunk = %d\n",s, grid.x, grid.y, grid.z, block.x, block.y, block.z, stream_chunk);
#ifdef USE_ASYNC
  	  GPUsequence<<<grid,block,0,streams[s]>>>(trk+(s*stream_chunk),hit+(s*stream_chunk*nlayer),outtrk+(s*stream_chunk),s);
#else
  	  GPUsequence<<<grid,block,0,0>>>(trk+(s*stream_chunk),hit+(s*stream_chunk*nlayer),outtrk+(s*stream_chunk),s);
#endif
    }  
    if(stream_remainder != 0){
#ifdef USE_ASYNC
  	  GPUsequenceR<<<grid,block,0,streams[num_streams]>>>(trk+(num_streams*stream_chunk),hit+(num_streams*stream_chunk*nlayer),outtrk+(num_streams*stream_chunk),num_streams);
#else
  	  GPUsequenceR<<<grid,block,0,0>>>(trk+(num_streams*stream_chunk),hit+(num_streams*stream_chunk*nlayer),outtrk+(num_streams*stream_chunk),num_streams);
#endif
    }
#ifdef include_data
	prefetch_host(outtrk, streams, stream_chunk, stream_remainder);
#endif
  } //end itr loop
  
  cudaDeviceSynchronize(); 
  auto wall_stop = std::chrono::high_resolution_clock::now();
#ifndef include_data
	prefetch_host(outtrk, streams, stream_chunk, stream_remainder);
#ifdef USE_ASYNC
    cudaDeviceSynchronize(); 
#endif
#endif

  for (int s = 0; s<stream_range;s++){
    cudaStreamDestroy(streams[s]);
  }
 
   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, num_streams);
#ifdef DUMP_OUTPUT
   FILE *fp_x;
   FILE *fp_y;
   FILE *fp_z;
   fp_x = fopen("output_x.txt", "w");
   fp_y = fopen("output_y.txt", "w");
   fp_z = fopen("output_z.txt", "w");
#endif

   int nnans = 0, nfail = 0;
   double avgx = 0, avgy = 0, avgz = 0;
   double avgpt = 0, avgphi = 0, avgtheta = 0;
   double avgdx = 0, avgdy = 0, avgdz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float pt_ = 1./ipt(outtrk,ie,it);
       float phi_ = phi(outtrk,ie,it);
       float theta_ = theta(outtrk,ie,it);
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (isnan(x_) ||
	   isnan(y_) ||
	   isnan(z_) ||
	   isnan(pt_) ||
	   isnan(phi_) ||
	   isnan(theta_)
	   ) {
	 nnans++;
	 continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
           fabs( (y_-hy_)/hy_ )>1. ||
           fabs( (z_-hz_)/hz_ )>1. ||
           fabs( (pt_-12.)/12.)>1.
           ) {
	 nfail++;
	 continue;
       }
#ifdef DUMP_OUTPUT
       fprintf(fp_x, "%f\n", x_);
       fprintf(fp_y, "%f\n", y_);
       fprintf(fp_z, "%f\n", z_);
#endif
       avgpt += pt_;
       avgphi += phi_;
       avgtheta += theta_;
       avgx += x_;
       avgy += y_;
       avgz += z_;
       avgdx += (x_-hx_)/x_;
       avgdy += (y_-hy_)/y_;
       avgdz += (z_-hz_)/z_;
     }
   }
#ifdef DUMP_OUTPUT
   fclose(fp_x);
   fclose(fp_y);
   fclose(fp_z);
   fp_x = fopen("input_x.txt", "w");
   fp_y = fopen("input_y.txt", "w");
   fp_z = fopen("input_z.txt", "w");
#endif
   avgpt = avgpt/double(nevts*ntrks);
   avgphi = avgphi/double(nevts*ntrks);
   avgtheta = avgtheta/double(nevts*ntrks);
   avgx = avgx/double(nevts*ntrks);
   avgy = avgy/double(nevts*ntrks);
   avgz = avgz/double(nevts*ntrks);
   avgdx = avgdx/double(nevts*ntrks);
   avgdy = avgdy/double(nevts*ntrks);
   avgdz = avgdz/double(nevts*ntrks);

   double stdx = 0, stdy = 0, stdz = 0;
   double stddx = 0, stddy = 0, stddz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
       float pt_ = 1./ipt(outtrk,ie,it);
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (isnan(x_) ||
	   isnan(y_) ||
	   isnan(z_)
	   ) {
	 continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
           fabs( (y_-hy_)/hy_ )>1. ||
           fabs( (z_-hz_)/hz_ )>1. ||
           fabs( (pt_-12.)/12.)>1.
           ) {
         continue;
       }
       stdx += (x_-avgx)*(x_-avgx);
       stdy += (y_-avgy)*(y_-avgy);
       stdz += (z_-avgz)*(z_-avgz);
       stddx += ((x_-hx_)/x_-avgdx)*((x_-hx_)/x_-avgdx);
       stddy += ((y_-hy_)/y_-avgdy)*((y_-hy_)/y_-avgdy);
       stddz += ((z_-hz_)/z_-avgdz)*((z_-hz_)/z_-avgdz);
#ifdef DUMP_OUTPUT
       x_ = x(trk,ie,it);
       y_ = y(trk,ie,it);
       z_ = z(trk,ie,it);
       fprintf(fp_x, "%f\n", x_);
       fprintf(fp_y, "%f\n", y_);
       fprintf(fp_z, "%f\n", z_);
#endif
     }
   }
#ifdef DUMP_OUTPUT
   fclose(fp_x);
   fclose(fp_y);
   fclose(fp_z);
#endif

   stdx = sqrtf(stdx/double(nevts*ntrks));
   stdy = sqrtf(stdy/double(nevts*ntrks));
   stdz = sqrtf(stdz/double(nevts*ntrks));
   stddx = sqrtf(stddx/double(nevts*ntrks));
   stddy = sqrtf(stddy/double(nevts*ntrks));
   stddz = sqrtf(stddz/double(nevts*ntrks));

   printf("track x avg=%f std/avg=%f\n", avgx, fabs(stdx/avgx));
   printf("track y avg=%f std/avg=%f\n", avgy, fabs(stdy/avgy));
   printf("track z avg=%f std/avg=%f\n", avgz, fabs(stdz/avgz));
   printf("track dx/x avg=%f std=%f\n", avgdx, stddx);
   printf("track dy/y avg=%f std=%f\n", avgdy, stddy);
   printf("track dz/z avg=%f std=%f\n", avgdz, stddz);
   printf("track pt avg=%f\n", avgpt);
   printf("track phi avg=%f\n", avgphi);
   printf("track theta avg=%f\n", avgtheta);
   printf("number of tracks with nans=%i\n", nnans);
   printf("number of tracks failed=%i\n", nfail);
	
   //free(trk);
   //free(hit);
   //free(outtrk);
   cudaFree(trk);
   cudaFree(hit);
   cudaFree(outtrk);
   
return 0;
}

