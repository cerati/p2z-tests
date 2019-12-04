/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/
//#include <cuda_profiler_api.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <unistd.h>
//#include <sys/time.h>
#ifndef _PROPAGATEGPU_
#define _PROPAGATEGPU_

#define nevts 1000
#define nb    600
#define bsize 16
#define ntrks nb*bsize
#define smear 0.1
#include "propagate-toz-test.h"

size_t GPUPosInMtrx(size_t i, size_t j, size_t D);
size_t GPUSymOffsets33(size_t i);
size_t GPUSymOffsets66(size_t i);



/*
__host__ __device__ struct ATRK {
  float par[6];
  float cov[21];
  int q;
  int hitidx[22];
};

__host__ __device__ struct AHIT {
  float pos[3];
  float cov[6];
};

__host__ __device__ struct MP1I {
  int data[1*bsize];
};

__host__ __device__ struct MP22I {
  int data[22*bsize];
};

__host__ __device__ struct MP3F {
  float data[3*bsize];
};

__host__ __device__ struct MP6F {
  float data[6*bsize];
};

__host__ __device__ struct MP3x3SF {
  float data[6*bsize];
};

__host__ __device__ struct MP6x6SF {
  float data[21*bsize];
};

__host__ __device__ struct MP6x6F {
  float data[36*bsize];
};

__host__ __device__ struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;
  MP22I   hitidx;
};

__host__ __device__ struct ALLTRKS {
  int ismade = 0;
  MPTRK  btrks[nevts*ntrks];
};

__host__ __device__ struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
};

__host__ __device__ struct ALLHITS {
  MPHIT bhits[nevts*ntrks];
};
*/

float GPUrandn(float mu, float sigma); 

MPTRK* bTk(ALLTRKS* tracks, size_t ev, size_t ib);
//MPTRK* GbTk(const ALLTRKS* tracks, size_t ev, size_t ib);
const MPTRK* bTk(const ALLTRKS* tracks, size_t ev, size_t ib);

const MPHIT* bHit(const ALLHITS* hits, size_t ev, size_t ib);
//MPHIT* GbHit(const ALLHITS* hits, size_t ev, size_t ib);
/*
float q(const MP1I* bq, size_t it);

float par(const MP6F* bpars, size_t it, size_t ipar);
float x    (const MP6F* bpars, size_t it);
float y    (const MP6F* bpars, size_t it);
float z    (const MP6F* bpars, size_t it);
float ipt  (const MP6F* bpars, size_t it);
float phi  (const MP6F* bpars, size_t it);
float theta(const MP6F* bpars, size_t it);

float par(const MPTRK* btracks, size_t it, size_t ipar);
float x    (const MPTRK* btracks, size_t it);
float y    (const MPTRK* btracks, size_t it);
float z    (const MPTRK* btracks, size_t it);
float ipt  (const MPTRK* btracks, size_t it);
float phi  (const MPTRK* btracks, size_t it);
float theta(const MPTRK* btracks, size_t it);

float par(const ALLTRKS* tracks, size_t ev, size_t tk, size_t ipar);
float x    (const ALLTRKS* tracks, size_t ev, size_t tk);
float y    (const ALLTRKS* tracks, size_t ev, size_t tk);
float z    (const ALLTRKS* tracks, size_t ev, size_t tk);
float ipt  (const ALLTRKS* tracks, size_t ev, size_t tk);
float phi  (const ALLTRKS* tracks, size_t ev, size_t tk);
float theta(const ALLTRKS* tracks, size_t ev, size_t tk);

void setpar(MP6F* bpars, size_t it, size_t ipar, float val);
void setx    (MP6F* bpars, size_t it, float val);
void sety    (MP6F* bpars, size_t it, float val);
void setz    (MP6F* bpars, size_t it, float val);
void setipt  (MP6F* bpars, size_t it, float val);
void setphi  (MP6F* bpars, size_t it, float val);
void settheta(MP6F* bpars, size_t it, float val);

void setpar(MPTRK* btracks, size_t it, size_t ipar, float val);
void setx    (MPTRK* btracks, size_t it, float val);
void sety    (MPTRK* btracks, size_t it, float val);
void setz    (MPTRK* btracks, size_t it, float val);
void setipt  (MPTRK* btracks, size_t it, float val);
void setphi  (MPTRK* btracks, size_t it, float val);
void settheta(MPTRK* btracks, size_t it, float val);


float pos(const MP3F* hpos, size_t it, size_t ipar);
float x(const MP3F* hpos, size_t it);
float y(const MP3F* hpos, size_t it);
float z(const MP3F* hpos, size_t it);
//
float pos(const MPHIT* hits, size_t it, size_t ipar);
float x(const MPHIT* hits, size_t it);
float y(const MPHIT* hits, size_t it);
float z(const MPHIT* hits, size_t it);

float pos(const ALLHITS* hits, size_t ev, size_t tk, size_t ipar);
float x(const ALLHITS* hits, size_t ev, size_t tk);
float y(const ALLHITS* hits, size_t ev, size_t tk);
float z(const ALLHITS* hits, size_t ev, size_t tk);
*/

//void GPUprepareTracks(ATRK inputtrk, ALLTRKS* result,const float* trkrandos1,const float* trkrandos2, const float* randoq);
//void GPUprepareHits(AHIT inputhit, ALLHITS *result,float* hitrandos1,float* hitrandos2);
//
////#define N bsize
//void GPUpropagateToZ(const MP6F* inPar,const MP1I* inChg, const MP3F* msP, MP6F* outPar, MP6x6F errorProp);
//void GPUMultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C);
//void GPUMultHelixPropTranspEndcap(const MP6x6F* A, const MP6x6F* B, MP6x6SF* C);
////__device__ MP6x6F errorProp, temp;

void allocateManaged( MPTRK* btracks, MPHIT* bhits, MPTRK* obtracks);
void allocateManagedx( ALLTRKS* trk_d, ALLHITS* hit_d, ALLTRKS* outtrk_d);
void allocateGPU(MPTRK* btracks_d, MPHIT* bhits_d, MPTRK* obtracks_d,MP6x6F errorProp_d, MP6x6F temp_d);
void cpyToGPU(const MPTRK* btracks, MPTRK* btracks_d,const MPHIT* bhits, MPHIT* bhits_d);
void cpyFromGPU(MPTRK* obtracks, MPTRK* obtracks_d);

//void GPUpropagateToZ(const MP6F* inPar,const MP1I* inChg, const MP3F* msP, MP6F* outPar, MP6x6F errorProp);
//
//void GPUtrackloop(const ALLTRKS* trk, const ALLHITS* hit, ALLTRKS* outtrk, int ie);
//void GPUeventloop(const ALLTRKS* trk, const ALLHITS* hit, ALLTRKS* outtrk);
void GPUSequence(const ALLTRKS* trk,const ALLHITS* hit, const ALLTRKS* outtrk,  size_t ie,  size_t ib);
//void GPUSequence(const ALLTRKS* trk,const ALLHITS* hit, const ALLTRKS* outtrk,  size_t ie,  size_t ib, MPTRK* btracks, MPHIT* bhits, MPTRK* obtracks);
//void GPUSequence(const MPTRK* btracks, const MPHIT* bhits, MPTRK* obtracks);
#endif
