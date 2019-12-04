/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/
//#include <cuda_profiler_api.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <unistd.h>
//#include <sys/time.h>
#ifndef _PROPAGATEGPUSTRUCTS_
#define _PROPAGATEGPUSTRUCTS_

#define nevts 1000
#define nb    600
#define bsize 16
#define ntrks nb*bsize
#define smear 0.1
#include "propagate-toz-test.h"

size_t GPUPosInMtrx(size_t i, size_t j, size_t D);
size_t GPUSymOffsets33(size_t i);
size_t GPUSymOffsets66(size_t i);




__host__ __device__ struct gATRK {
  float par[6];
  float cov[21];
  int q;
  int hitidx[22];
};

__host__ __device__ struct gAHIT {
  float pos[3];
  float cov[6];
};

__host__ __device__ struct gMP1I {
  int data[1*bsize];
};

__host__ __device__ struct gMP22I {
  int data[22*bsize];
};

__host__ __device__ struct gMP3F {
  float data[3*bsize];
};

__host__ __device__ struct gMP6F {
  float data[6*bsize];
};

__host__ __device__ struct gMP3x3SF {
  float data[6*bsize];
};

__host__ __device__ struct gMP6x6SF {
  float data[21*bsize];
};

__host__ __device__ struct gMP6x6F {
  float data[36*bsize];
};

__host__ __device__ struct gMPTRK {
  gMP6F    par;
  gMP6x6SF cov;
  gMP1I    q;
  gMP22I   hitidx;
};

__host__ __device__ struct gALLTRKS {
  gMPTRK  btrks[nevts*ntrks];
};

__host__ __device__ struct gMPHIT {
  gMP3F    pos;
  gMP3x3SF cov;
};

__host__ __device__ struct gALLHITS {
  gMPHIT bhits[nevts*ntrks];
};

float GPUrandn(float mu, float sigma); 

MPTRK* bTk(ALLTRKS* tracks, size_t ev, size_t ib);
gMPTRK* GbTk(const ALLTRKS* tracks, size_t ev, size_t ib);
const MPTRK* bTk(const ALLTRKS* tracks, size_t ev, size_t ib);

float q(const gMP1I* bq, size_t it);

float par(const gMP6F* bpars, size_t it, size_t ipar);
float x    (const gMP6F* bpars, size_t it);
float y    (const gMP6F* bpars, size_t it);
float z    (const gMP6F* bpars, size_t it);
float ipt  (const gMP6F* bpars, size_t it);
float phi  (const gMP6F* bpars, size_t it);
float theta(const gMP6F* bpars, size_t it);

float par(const gMPTRK* btracks, size_t it, size_t ipar);
float x    (const gMPTRK* btracks, size_t it);
float y    (const gMPTRK* btracks, size_t it);
float z    (const gMPTRK* btracks, size_t it);
float ipt  (const gMPTRK* btracks, size_t it);
float phi  (const gMPTRK* btracks, size_t it);
float theta(const gMPTRK* btracks, size_t it);

float par(const gALLTRKS* tracks, size_t ev, size_t tk, size_t ipar);
float x    (const gALLTRKS* tracks, size_t ev, size_t tk);
float y    (const gALLTRKS* tracks, size_t ev, size_t tk);
float z    (const gALLTRKS* tracks, size_t ev, size_t tk);
float ipt  (const gALLTRKS* tracks, size_t ev, size_t tk);
float phi  (const gALLTRKS* tracks, size_t ev, size_t tk);
float theta(const gALLTRKS* tracks, size_t ev, size_t tk);

void setpar(gMP6F* bpars, size_t it, size_t ipar, float val);
void setx    (gMP6F* bpars, size_t it, float val);
void sety    (gMP6F* bpars, size_t it, float val);
void setz    (gMP6F* bpars, size_t it, float val);
void setipt  (gMP6F* bpars, size_t it, float val);
void setphi  (gMP6F* bpars, size_t it, float val);
void settheta(gMP6F* bpars, size_t it, float val);

void setpar(gMPTRK* btracks, size_t it, size_t ipar, float val);
void setx    (gMPTRK* btracks, size_t it, float val);
void sety    (gMPTRK* btracks, size_t it, float val);
void setz    (gMPTRK* btracks, size_t it, float val);
void setipt  (gMPTRK* btracks, size_t it, float val);
void setphi  (gMPTRK* btracks, size_t it, float val);
void settheta(gMPTRK* btracks, size_t it, float val);

const MPHIT* bHit(const ALLHITS* hits, size_t ev, size_t ib);
gMPHIT* GbHit(const ALLHITS* hits, size_t ev, size_t ib);

float pos(const gMP3F* hpos, size_t it, size_t ipar);
float x(const gMP3F* hpos, size_t it);
float y(const gMP3F* hpos, size_t it);
float z(const gMP3F* hpos, size_t it);
//
float pos(const gMPHIT* hits, size_t it, size_t ipar);
float x(const gMPHIT* hits, size_t it);
float y(const gMPHIT* hits, size_t it);
float z(const gMPHIT* hits, size_t it);

float pos(const gALLHITS* hits, size_t ev, size_t tk, size_t ipar);
float x(const gALLHITS* hits, size_t ev, size_t tk);
float y(const gALLHITS* hits, size_t ev, size_t tk);
float z(const gALLHITS* hits, size_t ev, size_t tk);


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
void GPUSequence(const ALLTRKS* trk,const ALLHITS* hit, const ALLTRKS* outtrk,  size_t ie,  size_t ib, MPTRK* btracks, MPHIT* bhits, MPTRK* obtracks);
//void GPUSequence(const MPTRK* btracks, const MPHIT* bhits, MPTRK* obtracks);
#endif
