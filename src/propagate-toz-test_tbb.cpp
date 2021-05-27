/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#ifndef bsize
#define bsize 128
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    (ntrks/bsize)

#ifndef nevts
#define nevts 100
#endif
#define smear 0.00001

#ifndef NITER
#define NITER 5
#endif
#ifndef nlayer
#define nlayer 20
#endif

#ifndef nthreads
#define nthreads 64
#endif

using namespace tbb;

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

MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

int q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
float x    (const MP6F* bpars, size_t it){ return par(bpars, it, 0); }
float y    (const MP6F* bpars, size_t it){ return par(bpars, it, 1); }
float z    (const MP6F* bpars, size_t it){ return par(bpars, it, 2); }
float ipt  (const MP6F* bpars, size_t it){ return par(bpars, it, 3); }
float phi  (const MP6F* bpars, size_t it){ return par(bpars, it, 4); }
float theta(const MP6F* bpars, size_t it){ return par(bpars, it, 5); }
//
float par(const MPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
float x    (const MPTRK* btracks, size_t it){ return par(btracks, it, 0); }
float y    (const MPTRK* btracks, size_t it){ return par(btracks, it, 1); }
float z    (const MPTRK* btracks, size_t it){ return par(btracks, it, 2); }
float ipt  (const MPTRK* btracks, size_t it){ return par(btracks, it, 3); }
float phi  (const MPTRK* btracks, size_t it){ return par(btracks, it, 4); }
float theta(const MPTRK* btracks, size_t it){ return par(btracks, it, 5); }
//
float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
float x    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
float y    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
float z    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
float ipt  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
float phi  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
float theta(const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
void setx    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 0, val); }
void sety    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 1, val); }
void setz    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 2, val); }
void setipt  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 3, val); }
void setphi  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 4, val); }
void settheta(MP6F* bpars, size_t it, float val){ setpar(bpars, it, 5, val); }
//
void setpar(MPTRK* btracks, size_t it, size_t ipar, float val){
  setpar(&(*btracks).par,it,ipar,val);
}
void setx    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 0, val); }
void sety    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 1, val); }
void setz    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 2, val); }
void setipt  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 3, val); }
void setphi  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 4, val); }
void settheta(MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 5, val); }

const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib,size_t lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
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

MPTRK* prepareTracks(ATRK inputtrk) {
  MPTRK* result = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK)); //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
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

MPHIT* prepareHits(std::vector<AHIT>& inputhits) {
  MPHIT* result = (MPHIT*) malloc(nlayer*nevts*nb*sizeof(MPHIT));  //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {

    size_t mylay = lay;
    if (lay>=inputhits.size()) {
      // int wraplay = inputhits.size()/lay;
      exit(1);
    }
    AHIT& inputhit = inputhits[mylay];

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


#define N bsize
void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
//parallel_for(0,N,[&](int n){
#pragma omp simd 
 for (int n = 0; n < N; ++n)
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
  }//);
}

void MultHelixPropTranspEndcap(const MP6x6F* A, const MP6x6F* B, MP6x6SF* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
//parallel_for(0,N,[&](int n){
#pragma omp simd
  for (int n = 0; n < N; ++n)
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
  }//);
}


void KalmanGainInv(const MP6x6SF* A, const MP3x3SF* B, MP3x3* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#pragma omp simd
  for (int n = 0; n < N; ++n)
  {
    double det =
      ((a[0*N+n]+b[0*N+n])*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])))) -
      ((a[1*N+n]+b[1*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])))) +
      ((a[2*N+n]+b[2*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n]))));
    double invdet = 1.0/det;

    c[ 0*N+n] =  invdet*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 1*N+n] =  -1*invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 2*N+n] =  invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 3*N+n] =  -1*invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 4*N+n] =  invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 5*N+n] =  -1*invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 6*N+n] =  invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n])));
    c[ 7*N+n] =  -1*invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 8*N+n] =  invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[6*N+n]+b[3*N+n])) - ((a[1*N+n]+b[1*N+n]) *(a[1*N+n]+b[1*N+n])));
  }
}
void KalmanGain(const MP6x6SF* A, const MP3x3* B, MP3x6* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#pragma omp simd
  for (int n = 0; n < N; ++n)
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

void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){
  MP3x3 inverse_temp;
  MP3x6 kGain;
  MP6x6SF newErr;
  KalmanGainInv(trkErr,hitErr,&inverse_temp);
  KalmanGain(trkErr,&inverse_temp,&kGain);

#pragma omp simd
  for (size_t it=0;it<bsize;++it) {
    const float xin = x(inPar,it);
    const float yin = y(inPar,it);
    const float zin = z(inPar,it);
    const float ptin = 1./ipt(inPar,it);
    const float phiin = phi(inPar,it);
    const float thetain = theta(inPar,it);
    const float xout = x(msP,it);
    const float yout = y(msP,it);
    const float zout = z(msP,it);

    float xnew = xin + (kGain.data[0*bsize+it]*(xout-xin)) +(kGain.data[1*bsize+it]*(yout-yin)) +(kGain.data[2*bsize+it]*(zout-zin));
    float ynew = yin + (kGain.data[3*bsize+it]*(xout-xin)) +(kGain.data[4*bsize+it]*(yout-yin)) +(kGain.data[5*bsize+it]*(zout-zin));
    float znew = zin + (kGain.data[6*bsize+it]*(xout-xin)) +(kGain.data[7*bsize+it]*(yout-yin)) +(kGain.data[8*bsize+it]*(zout-zin));
    float ptnew = ptin + (kGain.data[9*bsize+it]*(xout-xin)) +(kGain.data[10*bsize+it]*(yout-yin)) +(kGain.data[11*bsize+it]*(zout-zin));
    float phinew = phiin + (kGain.data[12*bsize+it]*(xout-xin)) +(kGain.data[13*bsize+it]*(yout-yin)) +(kGain.data[14*bsize+it]*(zout-zin));
    float thetanew = thetain + (kGain.data[15*bsize+it]*(xout-xin)) +(kGain.data[16*bsize+it]*(yout-yin)) +(kGain.data[17*bsize+it]*(zout-zin));

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
  (*trkErr) = newErr;
}

void KalmanUpdate_v2(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){

   // AddIntoUpperLeft2x2(psErr, msErr, resErr);
   MP2x2SF resErr_loc;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
     resErr_loc.data[0*bsize+it] = trkErr->data[0*bsize+it] + hitErr->data[0*bsize+it];
     resErr_loc.data[1*bsize+it] = trkErr->data[1*bsize+it] + hitErr->data[1*bsize+it];
     resErr_loc.data[2*bsize+it] = trkErr->data[2*bsize+it] + hitErr->data[2*bsize+it];
   }

   // Matriplex::InvertCramerSym(resErr);
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
     const double det = (double)resErr_loc.data[0*bsize+it] * resErr_loc.data[2*bsize+it] -
                        (double)resErr_loc.data[1*bsize+it] * resErr_loc.data[1*bsize+it];
     const float s   = 1.f / det;
     const float tmp = s * resErr_loc.data[2*bsize+it];
     resErr_loc.data[1*bsize+it] *= -s;
     resErr_loc.data[2*bsize+it]  = s * resErr_loc.data[0*bsize+it];
     resErr_loc.data[0*bsize+it]  = tmp;
   }

   // KalmanGain(psErr, resErr, K);
   MP2x6 kGain;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
      kGain.data[ 0*bsize+it] = trkErr->data[ 0*bsize+it]*resErr_loc.data[ 0*bsize+it] + trkErr->data[ 1*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 1*bsize+it] = trkErr->data[ 0*bsize+it]*resErr_loc.data[ 1*bsize+it] + trkErr->data[ 1*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 2*bsize+it] = trkErr->data[ 1*bsize+it]*resErr_loc.data[ 0*bsize+it] + trkErr->data[ 2*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 3*bsize+it] = trkErr->data[ 1*bsize+it]*resErr_loc.data[ 1*bsize+it] + trkErr->data[ 2*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 4*bsize+it] = trkErr->data[ 3*bsize+it]*resErr_loc.data[ 0*bsize+it] + trkErr->data[ 4*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 5*bsize+it] = trkErr->data[ 3*bsize+it]*resErr_loc.data[ 1*bsize+it] + trkErr->data[ 4*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 6*bsize+it] = trkErr->data[ 6*bsize+it]*resErr_loc.data[ 0*bsize+it] + trkErr->data[ 7*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 7*bsize+it] = trkErr->data[ 6*bsize+it]*resErr_loc.data[ 1*bsize+it] + trkErr->data[ 7*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 8*bsize+it] = trkErr->data[10*bsize+it]*resErr_loc.data[ 0*bsize+it] + trkErr->data[11*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 9*bsize+it] = trkErr->data[10*bsize+it]*resErr_loc.data[ 1*bsize+it] + trkErr->data[11*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[10*bsize+it] = trkErr->data[15*bsize+it]*resErr_loc.data[ 0*bsize+it] + trkErr->data[16*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[11*bsize+it] = trkErr->data[15*bsize+it]*resErr_loc.data[ 1*bsize+it] + trkErr->data[16*bsize+it]*resErr_loc.data[ 2*bsize+it];
   }

   // SubtractFirst2(msPar, psPar, res);
   // MultResidualsAdd(K, psPar, res, outPar);
   MP2F res_loc;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
     res_loc.data[0*bsize+it] =  x(msP,it) - x(inPar,it);
     res_loc.data[1*bsize+it] =  y(msP,it) - y(inPar,it);

     setx    (inPar, it, x    (inPar, it) + kGain.data[ 0*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 1*bsize+it] * res_loc.data[ 1*bsize+it]);
     sety    (inPar, it, y    (inPar, it) + kGain.data[ 2*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 3*bsize+it] * res_loc.data[ 1*bsize+it]);
     setz    (inPar, it, z    (inPar, it) + kGain.data[ 4*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 5*bsize+it] * res_loc.data[ 1*bsize+it]);
     setipt  (inPar, it, ipt  (inPar, it) + kGain.data[ 6*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 7*bsize+it] * res_loc.data[ 1*bsize+it]);
     setphi  (inPar, it, phi  (inPar, it) + kGain.data[ 8*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 9*bsize+it] * res_loc.data[ 1*bsize+it]);
     settheta(inPar, it, theta(inPar, it) + kGain.data[10*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[11*bsize+it] * res_loc.data[ 1*bsize+it]);
     //note: if ipt changes sign we should update the charge, or we should get rid of the charge altogether and just use the sign of ipt
   }

   // squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
   // missing

   // KHC(K, psErr, outErr);
   // outErr.Subtract(psErr, outErr);
   MP6x6SF newErr;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
      newErr.data[ 0*bsize+it] = kGain.data[ 0*bsize+it]*trkErr->data[ 0*bsize+it] + kGain.data[ 1*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr.data[ 1*bsize+it] = kGain.data[ 2*bsize+it]*trkErr->data[ 0*bsize+it] + kGain.data[ 3*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr.data[ 2*bsize+it] = kGain.data[ 2*bsize+it]*trkErr->data[ 1*bsize+it] + kGain.data[ 3*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr.data[ 3*bsize+it] = kGain.data[ 4*bsize+it]*trkErr->data[ 0*bsize+it] + kGain.data[ 5*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr.data[ 4*bsize+it] = kGain.data[ 4*bsize+it]*trkErr->data[ 1*bsize+it] + kGain.data[ 5*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr.data[ 5*bsize+it] = kGain.data[ 4*bsize+it]*trkErr->data[ 3*bsize+it] + kGain.data[ 5*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr.data[ 6*bsize+it] = kGain.data[ 6*bsize+it]*trkErr->data[ 0*bsize+it] + kGain.data[ 7*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr.data[ 7*bsize+it] = kGain.data[ 6*bsize+it]*trkErr->data[ 1*bsize+it] + kGain.data[ 7*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr.data[ 8*bsize+it] = kGain.data[ 6*bsize+it]*trkErr->data[ 3*bsize+it] + kGain.data[ 7*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr.data[ 9*bsize+it] = kGain.data[ 6*bsize+it]*trkErr->data[ 6*bsize+it] + kGain.data[ 7*bsize+it]*trkErr->data[ 7*bsize+it];
      newErr.data[10*bsize+it] = kGain.data[ 8*bsize+it]*trkErr->data[ 0*bsize+it] + kGain.data[ 9*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr.data[11*bsize+it] = kGain.data[ 8*bsize+it]*trkErr->data[ 1*bsize+it] + kGain.data[ 9*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr.data[12*bsize+it] = kGain.data[ 8*bsize+it]*trkErr->data[ 3*bsize+it] + kGain.data[ 9*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr.data[13*bsize+it] = kGain.data[ 8*bsize+it]*trkErr->data[ 6*bsize+it] + kGain.data[ 9*bsize+it]*trkErr->data[ 7*bsize+it];
      newErr.data[14*bsize+it] = kGain.data[ 8*bsize+it]*trkErr->data[10*bsize+it] + kGain.data[ 9*bsize+it]*trkErr->data[11*bsize+it];
      newErr.data[15*bsize+it] = kGain.data[10*bsize+it]*trkErr->data[ 0*bsize+it] + kGain.data[11*bsize+it]*trkErr->data[ 1*bsize+it];
      newErr.data[16*bsize+it] = kGain.data[10*bsize+it]*trkErr->data[ 1*bsize+it] + kGain.data[11*bsize+it]*trkErr->data[ 2*bsize+it];
      newErr.data[17*bsize+it] = kGain.data[10*bsize+it]*trkErr->data[ 3*bsize+it] + kGain.data[11*bsize+it]*trkErr->data[ 4*bsize+it];
      newErr.data[18*bsize+it] = kGain.data[10*bsize+it]*trkErr->data[ 6*bsize+it] + kGain.data[11*bsize+it]*trkErr->data[ 7*bsize+it];
      newErr.data[19*bsize+it] = kGain.data[10*bsize+it]*trkErr->data[10*bsize+it] + kGain.data[11*bsize+it]*trkErr->data[11*bsize+it];
      newErr.data[20*bsize+it] = kGain.data[10*bsize+it]*trkErr->data[15*bsize+it] + kGain.data[11*bsize+it]*trkErr->data[16*bsize+it];

      newErr.data[ 0*bsize+it] = trkErr->data[ 0*bsize+it] - newErr.data[ 0*bsize+it];
      newErr.data[ 1*bsize+it] = trkErr->data[ 1*bsize+it] - newErr.data[ 1*bsize+it];
      newErr.data[ 2*bsize+it] = trkErr->data[ 2*bsize+it] - newErr.data[ 2*bsize+it];
      newErr.data[ 3*bsize+it] = trkErr->data[ 3*bsize+it] - newErr.data[ 3*bsize+it];
      newErr.data[ 4*bsize+it] = trkErr->data[ 4*bsize+it] - newErr.data[ 4*bsize+it];
      newErr.data[ 5*bsize+it] = trkErr->data[ 5*bsize+it] - newErr.data[ 5*bsize+it];
      newErr.data[ 6*bsize+it] = trkErr->data[ 6*bsize+it] - newErr.data[ 6*bsize+it];
      newErr.data[ 7*bsize+it] = trkErr->data[ 7*bsize+it] - newErr.data[ 7*bsize+it];
      newErr.data[ 8*bsize+it] = trkErr->data[ 8*bsize+it] - newErr.data[ 8*bsize+it];
      newErr.data[ 9*bsize+it] = trkErr->data[ 9*bsize+it] - newErr.data[ 9*bsize+it];
      newErr.data[10*bsize+it] = trkErr->data[10*bsize+it] - newErr.data[10*bsize+it];
      newErr.data[11*bsize+it] = trkErr->data[11*bsize+it] - newErr.data[11*bsize+it];
      newErr.data[12*bsize+it] = trkErr->data[12*bsize+it] - newErr.data[12*bsize+it];
      newErr.data[13*bsize+it] = trkErr->data[13*bsize+it] - newErr.data[13*bsize+it];
      newErr.data[14*bsize+it] = trkErr->data[14*bsize+it] - newErr.data[14*bsize+it];
      newErr.data[15*bsize+it] = trkErr->data[15*bsize+it] - newErr.data[15*bsize+it];
      newErr.data[16*bsize+it] = trkErr->data[16*bsize+it] - newErr.data[16*bsize+it];
      newErr.data[17*bsize+it] = trkErr->data[17*bsize+it] - newErr.data[17*bsize+it];
      newErr.data[18*bsize+it] = trkErr->data[18*bsize+it] - newErr.data[18*bsize+it];
      newErr.data[19*bsize+it] = trkErr->data[19*bsize+it] - newErr.data[19*bsize+it];
      newErr.data[20*bsize+it] = trkErr->data[20*bsize+it] - newErr.data[20*bsize+it];
   }

   (*trkErr) = newErr;
}

const float kfact= 100/(-0.299792458*3.8112);
void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar, const MP1I* inChg, 
                  const MP3F* msP, MP6x6SF* outErr, MP6F* outPar) {
  
  MP6x6F errorProp, temp;
#pragma omp simd
  for (size_t it=0;it<bsize;++it) {	
    const float zout = z(msP,it);
    //note: in principle charge is not needed and could be the sign of ipt
    const float k = q(inChg,it)*kfact;
    const float deltaZ = zout - z(inPar,it);
    const float pt = 1./ipt(inPar,it);
    const float cosP = cosf(phi(inPar,it));
    const float sinP = sinf(phi(inPar,it));
    const float cosT = cosf(theta(inPar,it));
    const float sinT = sinf(theta(inPar,it));
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*ipt(inPar,it)*icosTk;
    //const float alpha = deltaZ*sinT*ipt(inPar,it)/(cosT*k);
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;
    setx(outPar,it, x(inPar,it) + k*(pxin*sina - pyin*(1.-cosa)) );
    sety(outPar,it, y(inPar,it) + k*(pyin*sina + pxin*(1.-cosa)) );
    setz(outPar,it,zout);
    setipt(outPar,it, ipt(inPar,it));
    setphi(outPar,it, phi(inPar,it)+alpha );
    settheta(outPar,it, theta(inPar,it) );
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
    
    for (size_t i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.;
    errorProp.data[bsize*PosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp.data[bsize*PosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.-cCosPsina))*(pt*pt);
    errorProp.data[bsize*PosInMtrx(0,4,6) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp.data[bsize*PosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*icosT);
    errorProp.data[bsize*PosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp.data[bsize*PosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.-cCosPsina))*(pt*pt);
    errorProp.data[bsize*PosInMtrx(1,4,6) + it] = (k*pt)*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp.data[bsize*PosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp.data[bsize*PosInMtrx(4,2,6) + it] = -ipt(inPar,it)*sinT*(icosTk);
    errorProp.data[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ*(icosTk);
    errorProp.data[bsize*PosInMtrx(4,5,6) + it] = ipt(inPar,it)*deltaZ*(icosT*icosTk);
//    errorProp.data[bsize*PosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
//    errorProp.data[bsize*PosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*ipt(inPar,it))-k*(cosP*sina-sinP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
//    errorProp.data[bsize*PosInMtrx(0,4,6) + it] = (k/ipt(inPar,it))*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
//    errorProp.data[bsize*PosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
//    errorProp.data[bsize*PosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
//    errorProp.data[bsize*PosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*ipt(inPar,it))-k*(sinP*sina+cosP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
//    errorProp.data[bsize*PosInMtrx(1,4,6) + it] = (k/ipt(inPar,it))*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
//    errorProp.data[bsize*PosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
//    errorProp.data[bsize*PosInMtrx(4,2,6) + it] = -ipt(inPar,it)*sinT/(cosT*k);
//    errorProp.data[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ/(cosT*k);
//    errorProp.data[bsize*PosInMtrx(4,5,6) + it] = ipt(inPar,it)*deltaZ/(cosT*cosT*k);
  }
  
  MultHelixPropEndcap(&errorProp, inErr, &temp);
  MultHelixPropTranspEndcap(&errorProp, &temp, outErr);
}

int main (int argc, char* argv[]) {

#include "input_track.h"

  std::vector<AHIT> inputhits{inputhit25,inputhit24,inputhit23,inputhit22,inputhit21,inputhit20,inputhit19,inputhit18,inputhit17,
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
   MPTRK* trk = prepareTracks(inputtrk);
   MPHIT* hit = prepareHits(inputhits);
   MPTRK* outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");
   
   global_control c(global_control::max_allowed_parallelism, nthreads);
   //task_scheduler_init init(nthreads);

   auto wall_start = std::chrono::high_resolution_clock::now();

   int itr;
   for(itr=0; itr<NITER; itr++) {
      parallel_for(blocked_range<size_t>(0,nevts,4),[&](blocked_range<size_t> iex){
      for(size_t ie =iex.begin(); ie<iex.end();++ie){
        parallel_for(blocked_range<size_t>(0,nb,4),[&](blocked_range<size_t> ibx){
        for(size_t ib =ibx.begin(); ib<ibx.end();++ib){
          const MPTRK* btracks = bTk(trk, ie, ib);
          MPTRK* obtracks = bTk(outtrk, ie, ib);
	  (*obtracks) = (*btracks);
          for(size_t layer=0; layer<nlayer; ++layer) {
            const MPHIT* bhits = bHit(hit, ie, ib,layer);
            propagateToZ(&(*obtracks).cov, &(*obtracks).par, &(*obtracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
            KalmanUpdate(&(*obtracks).cov, &(*obtracks).par, &(*bhits).cov,  &(*bhits).pos);//use v2 for better physics performance
            //KalmanUpdate_v2(&(*obtracks).cov, &(*obtracks).par, &(*bhits).cov,  &(*bhits).pos);
          }
        }});
      }});
   } //end of itr loop
   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, nthreads);

   int nnans = 0, nfail = 0;
   float avgx = 0, avgy = 0, avgz = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float pt_ = 1./ipt(outtrk,ie,it);
       float phi_ = phi(outtrk,ie,it);
       float theta_ = theta(outtrk,ie,it);
       float hx_ = inputhits[nlayer-1].pos[0];
       float hy_ = inputhits[nlayer-1].pos[1];
       float hz_ = inputhits[nlayer-1].pos[2];
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
	   std::isfinite(y_)==false ||
	   std::isfinite(z_)==false ||
	   std::isfinite(pt_)==false ||
	   std::isfinite(phi_)==false ||
	   std::isfinite(theta_)==false
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
   avgpt = avgpt/float(nevts*ntrks);
   avgphi = avgphi/float(nevts*ntrks);
   avgtheta = avgtheta/float(nevts*ntrks);
   avgx = avgx/float(nevts*ntrks);
   avgy = avgy/float(nevts*ntrks);
   avgz = avgz/float(nevts*ntrks);
   avgdx = avgdx/float(nevts*ntrks);
   avgdy = avgdy/float(nevts*ntrks);
   avgdz = avgdz/float(nevts*ntrks);

   float stdx = 0, stdy = 0, stdz = 0;
   float stddx = 0, stddy = 0, stddz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float pt_ = 1./ipt(outtrk,ie,it);
       float hx_ = inputhits[nlayer-1].pos[0];
       float hy_ = inputhits[nlayer-1].pos[1];
       float hz_ = inputhits[nlayer-1].pos[2];
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
	   std::isfinite(y_)==false ||
	   std::isfinite(z_)==false
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
     }
   }

   stdx = sqrtf(stdx/float(nevts*ntrks));
   stdy = sqrtf(stdy/float(nevts*ntrks));
   stdz = sqrtf(stdz/float(nevts*ntrks));
   stddx = sqrtf(stddx/float(nevts*ntrks));
   stddy = sqrtf(stddy/float(nevts*ntrks));
   stddz = sqrtf(stddz/float(nevts*ntrks));

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

   free(trk);
   free(hit);
   free(outtrk);

   return 0;
}
