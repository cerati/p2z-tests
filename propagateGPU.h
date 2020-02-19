/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/
//#include <cuda_profiler_api.h>
//#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#ifndef _PROPAGATEGPU_
#define _PROPAGATEGPU_

#define nevts 100
#define nb    600
#define bsize 16
#define ntrks nb*bsize
#define smear 0.1
//#include "propagate-toz-test.h"
#include "cuda_profiler_api.h"

#if USE_ACC
#include <accelmath.h>
#endif

#if USE_GPU
#define HOSTDEV __host__  __device__
#else
#define HOSTDEV
#endif

HOSTDEV size_t GPUPosInMtrx(size_t i, size_t j, size_t D);
HOSTDEV size_t GPUSymOffsets33(size_t i);
HOSTDEV size_t GPUSymOffsets66(size_t i);
HOSTDEV size_t PosInMtrx(size_t i, size_t j, size_t D);
HOSTDEV size_t SymOffsets33(size_t i);
HOSTDEV size_t SymOffsets66(size_t i);

float randn(float mu,float sigma);

struct ATRK {
  float par[6];
  float cov[21];
  //float* par = new float[6];
  //float* cov= new float[21];
  int q;
  //int* hitidx = new int[22];
  int hitidx[22];
};

struct AHIT {
  //float* pos= new float[3];
  //float* cov = new float[6];
  float pos[3];
  float cov[6];
};

struct MP1I {
//  int* data= new int[1*bsize];
  int data[1*bsize];
};

struct MP22I {
  //int* data= new int[22*bsize];
  int data[22*bsize];
};

struct MP3F {
  //float* data = new float[3*bsize];
  float data[3*bsize];
};

struct MP6F {
  //float* data = new float[6*bsize];
  float data[6*bsize];
};

struct MP3x3SF {
  //float* data = new float[6*bsize];
  float data[6*bsize];
};

struct MP6x6SF {
  //float* data = new float[21*bsize];
  float data[21*bsize];
};

struct MP6x6F {
  float data[36*bsize];
  //float* data = new float[36*bsize];
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;
  MP22I   hitidx;
};

struct ALLTRKS {
  //int ismade = 0;
  //MPTRK*  btrks= new MPTRK[nevts*ntrks];
  MPTRK  btrks[nevts*ntrks];
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
};

struct ALLHITS {
  //MPHIT* bhits = new MPHIT[nevts*ntrks];
  MPHIT bhits[nevts*ntrks];
};


float GPUrandn(float mu, float sigma); 

ALLTRKS* prepareTracks(ATRK inputtrk);
ALLHITS* prepareHits(AHIT inputhit);

HOSTDEV MPTRK* bTk(ALLTRKS* tracks, size_t ev, size_t ib);
HOSTDEV const MPTRK* bTk(const ALLTRKS* tracks, size_t ev, size_t ib);

HOSTDEV const MPHIT* bHit(const ALLHITS* hits, size_t ev, size_t ib);

HOSTDEV float q(const MP1I* bq, size_t it);

HOSTDEV float par(const MP6F* bpars, size_t it, size_t ipar);
HOSTDEV float x    (const MP6F* bpars, size_t it);
HOSTDEV float y    (const MP6F* bpars, size_t it);
HOSTDEV float z    (const MP6F* bpars, size_t it);
HOSTDEV float ipt  (const MP6F* bpars, size_t it);
HOSTDEV float phi  (const MP6F* bpars, size_t it);
HOSTDEV float theta(const MP6F* bpars, size_t it);

HOSTDEV float par(const MPTRK* btracks, size_t it, size_t ipar);
HOSTDEV float x    (const MPTRK* btracks, size_t it);
HOSTDEV float y    (const MPTRK* btracks, size_t it);
HOSTDEV float z    (const MPTRK* btracks, size_t it);
HOSTDEV float ipt  (const MPTRK* btracks, size_t it);
HOSTDEV float phi  (const MPTRK* btracks, size_t it);
HOSTDEV float theta(const MPTRK* btracks, size_t it);

HOSTDEV float par(const ALLTRKS* tracks, size_t ev, size_t tk, size_t ipar);
HOSTDEV float x    (const ALLTRKS* tracks, size_t ev, size_t tk);
HOSTDEV float y    (const ALLTRKS* tracks, size_t ev, size_t tk);
HOSTDEV float z    (const ALLTRKS* tracks, size_t ev, size_t tk);
HOSTDEV float ipt  (const ALLTRKS* tracks, size_t ev, size_t tk);
HOSTDEV float phi  (const ALLTRKS* tracks, size_t ev, size_t tk);
HOSTDEV float theta(const ALLTRKS* tracks, size_t ev, size_t tk);

HOSTDEV void setpar(MP6F* bpars, size_t it, size_t ipar, float val);
HOSTDEV void setx    (MP6F* bpars, size_t it, float val);
HOSTDEV void sety    (MP6F* bpars, size_t it, float val);
HOSTDEV void setz    (MP6F* bpars, size_t it, float val);
HOSTDEV void setipt  (MP6F* bpars, size_t it, float val);
HOSTDEV void setphi  (MP6F* bpars, size_t it, float val);
HOSTDEV void settheta(MP6F* bpars, size_t it, float val);

HOSTDEV void setpar(MPTRK* btracks, size_t it, size_t ipar, float val);
HOSTDEV void setx    (MPTRK* btracks, size_t it, float val);
HOSTDEV void sety    (MPTRK* btracks, size_t it, float val);
HOSTDEV void setz    (MPTRK* btracks, size_t it, float val);
HOSTDEV void setipt  (MPTRK* btracks, size_t it, float val);
HOSTDEV void setphi  (MPTRK* btracks, size_t it, float val);
HOSTDEV void settheta(MPTRK* btracks, size_t it, float val);


HOSTDEV float pos(const MP3F* hpos, size_t it, size_t ipar);
HOSTDEV float x(const MP3F* hpos, size_t it);
HOSTDEV float y(const MP3F* hpos, size_t it);
HOSTDEV float z(const MP3F* hpos, size_t it);

HOSTDEV float pos(const MPHIT* hits, size_t it, size_t ipar);
HOSTDEV float x(const MPHIT* hits, size_t it);
HOSTDEV float y(const MPHIT* hits, size_t it);
HOSTDEV float z(const MPHIT* hits, size_t it);

HOSTDEV float pos(const ALLHITS* hits, size_t ev, size_t tk, size_t ipar);
HOSTDEV float x(const ALLHITS* hits, size_t ev, size_t tk);
HOSTDEV float y(const ALLHITS* hits, size_t ev, size_t tk);
HOSTDEV float z(const ALLHITS* hits, size_t ev, size_t tk);



HOSTDEV void allocateManaged( MPTRK* btracks, MPHIT* bhits, MPTRK* obtracks);
HOSTDEV void allocateManagedx( ALLTRKS* trk_d, ALLHITS* hit_d, ALLTRKS* outtrk_d);
HOSTDEV void allocateGPU(const MPTRK* btracks_d, const MPHIT* bhits_d, MPTRK* obtracks_d,MP6x6F errorProp_d, MP6x6F temp_d);
HOSTDEV void cpyToGPU(const MPTRK* btracks, MPTRK* btracks_d,const MPHIT* bhits, MPHIT* bhits_d);
HOSTDEV void cpyFromGPU(MPTRK* obtracks, MPTRK* obtracks_d);

HOSTDEV void GPUsequence1(ALLTRKS* trk, ALLHITS* hit, ALLTRKS* outtrk,cudaStream_t* streams);
__global__ void GPUsequence(ALLTRKS* trk, ALLHITS* hit, ALLTRKS* outtrk,int streams);

HOSTDEV void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C);
HOSTDEV void MultHelixPropTranspEndcap(const MP6x6F* A, const MP6x6F* B, MP6x6SF* C);
HOSTDEV void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar,
                  const MP1I* inChg, const MP3F* msP,
                        MP6x6SF* outErr, MP6F* outPar);

#endif
