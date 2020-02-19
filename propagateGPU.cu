/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/
#include <cuda_profiler_api.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "propagateGPU.h"
//#include "propagateGPUStructs.cuh"
//#include "propagate-toz-test.h"

//#define nevts 100
//#define nb    600
//#define bsize 16
//#define ntrks nb*bsize
//#define smear 0.1


//#if USE_GPU
//#define HostDEV __host__ __device__
//#else
//#define HOSTDEV
//#endif


HOSTDEV size_t GPUPosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

HOSTDEV size_t GPUSymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

HOSTDEV size_t GPUSymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}

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




ALLTRKS* prepareTracks(ATRK inputtrk) {
#if USE_GPU
  ALLTRKS* result;
  cudaMallocManaged((void**)&result,sizeof(ALLTRKS)); //fixme, align?
#else
  ALLTRKS* result = (ALLTRKS*) malloc(sizeof(ALLTRKS)); //fixme, align?
#endif
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //par
        for (size_t ip=0;ip<6;++ip) {
          (*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
        }
        //cov
        for (size_t ip=0;ip<36;++ip) {
          (*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
        }
        //q
        (*result).btrks[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      }
    }
  }
  return result;
}

ALLHITS* prepareHits(AHIT inputhit) {
#if USE_GPU
  ALLHITS* result;
  cudaMallocManaged((void**)&result,sizeof(ALLHITS));  //fixme, align?
#else
  //ALLHITS* result = (ALLHITS*) malloc(sizeof(ALLHITS));  //fixme, align?
#endif
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //pos
        for (size_t ip=0;ip<3;++ip) {
          (*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        }
        //cov
        for (size_t ip=0;ip<6;++ip) {
          (*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        }
      }
    }
  }
  return result;
}



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


float GPUrandn(float mu, float sigma) {
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

HOSTDEV MPTRK* bTk(ALLTRKS* tracks, size_t ev, size_t ib) {
  return &((*tracks).btrks[ib + nb*ev]);
}

HOSTDEV const MPTRK* bTk(const ALLTRKS* tracks, size_t ev, size_t ib) {
  return &((*tracks).btrks[ib + nb*ev]);
}
HOSTDEV MPTRK GbTk(ALLTRKS* tracks, size_t ev, size_t ib) {
  return ((*tracks).btrks[ib + nb*ev]);
}


HOSTDEV float q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
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
//
HOSTDEV float par(const ALLTRKS* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
HOSTDEV float x    (const ALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
HOSTDEV float y    (const ALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
HOSTDEV float z    (const ALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
HOSTDEV float ipt  (const ALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
HOSTDEV float phi  (const ALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
HOSTDEV float theta(const ALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
HOSTDEV void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
HOSTDEV void setx    (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 0, val); }
HOSTDEV void sety    (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 1, val); }
HOSTDEV void setz    (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 2, val); }
HOSTDEV void setipt  (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 3, val); }
HOSTDEV void setphi  (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 4, val); }
HOSTDEV void settheta(MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 5, val); }
//
HOSTDEV void setpar(MPTRK* btracks, size_t it, size_t ipar, float val){
  return setpar(&(*btracks).par,it,ipar,val);
}
HOSTDEV void setx    (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 0, val); }
HOSTDEV void sety    (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 1, val); }
HOSTDEV void setz    (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 2, val); }
HOSTDEV void setipt  (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 3, val); }
HOSTDEV void setphi  (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 4, val); }
HOSTDEV void settheta(MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 5, val); }

HOSTDEV MPHIT* bHit(ALLHITS* hits, size_t ev, size_t ib) {
  return &((*hits).bhits[ib + nb*ev]);
}
HOSTDEV const MPHIT* bHit(const ALLHITS* hits, size_t ev, size_t ib) {
  return &((*hits).bhits[ib + nb*ev]);
}
//
HOSTDEV float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
HOSTDEV float x(const MP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
HOSTDEV float y(const MP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
HOSTDEV float z(const MP3F* hpos, size_t it)    { return pos(hpos, it, 2); }
//
HOSTDEV float pos(const MPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t it)    { return pos(hits, it, 0); }
HOSTDEV float y(const MPHIT* hits, size_t it)    { return pos(hits, it, 1); }
HOSTDEV float z(const MPHIT* hits, size_t it)    { return pos(hits, it, 2); }
//
HOSTDEV float pos(const ALLHITS* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
HOSTDEV float x(const ALLHITS* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
HOSTDEV float y(const ALLHITS* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
HOSTDEV float z(const ALLHITS* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }


//__global__ 
//void GPUprepareTracks(ATRK inputtrk, ALLTRKS* result,const float* trkrandos1,const float* trkrandos2, const float* randoq) {
//  
//  // store in element order for bunches of bsize matrices (a la matriplex) 
//  printf("par: %f,%f,%f,%f,%f,%f\n",inputtrk.par[0],inputtrk.par[1],inputtrk.par[2],inputtrk.par[3],inputtrk.par[4],inputtrk.par[5]);
// 
//  for (size_t ie=0;ie<nevts;++ie) {
//    for (size_t ib=0;ib<nb;++ib) {
//      for (size_t it=0;it<bsize;++it) {
//	for (size_t ip=0;ip<6;++ip) {
//	 // (*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*2)*inputtrk.par[ip];
//	  (*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*trkrandos1[ie+ib*nevts+it*nb+ip*bsize])*inputtrk.par[ip];
//	  //(*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
//	}
//	//cov
//	for (size_t ip=0;ip<36;++ip) {
//	  //(*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*2)*inputtrk.cov[ip];
//	  (*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*trkrandos2[ie+ib*nevts+it*nb+ip*bsize])*inputtrk.cov[ip];
//	  //(*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
//	}
//	//q
//	(*result).btrks[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + randoq[0]);//fixme check
//      }
//    }
//  } 
//  // printf("results:");// %f\n",(*result).btrks[0].par.data[0]);
//  //outtrk = result;
//}
//
//__global__ 
//void GPUprepareHits(AHIT inputhit, ALLHITS *result,float* hitrandos1,float* hitrandos2) {
//  //ALLHITS* result = (ALLHITS*) malloc(sizeof(ALLHITS));  //fixme, align?
//  // store in element order for bunches of bsize matrices (a la matriplex)
//  for (size_t ie=0;ie<nevts;++ie) {
//    for (size_t ib=0;ib<nb;++ib) {
//      for (size_t it=0;it<bsize;++it) {
//  	//pos
//  	for (size_t ip=0;ip<3;++ip) {
//  	  //(*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*2)*inputhit.pos[ip];
//  	  (*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*hitrandos1[ie+ib*nevts+it*nb+ip*bsize])*inputhit.pos[ip];
//  	  //(*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
//  	}
//  	//cov
//  	for (size_t ip=0;ip<6;++ip) {
//  	  //(*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*2)*inputhit.cov[ip];
//  	  (*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*hitrandos2[ie+ib*nevts+it*nb+ip*bsize])*inputhit.cov[ip];
//  	  //(*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
//  	}
//      }
//    }
//  }
//  //outtrk = result;
//}

#define N bsize
//__device__ 
//void GPUMultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
HOSTDEV void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#if USE_ACC
#else
#if USE_GPU
#else
#pragma omp simd
#endif
#endif
  //int n = threadIdx.x + blockIdx.x*blockDim.x;
  for (int n = 0; n < N; ++n)
//while(n<N)
  {
  //printf("testing: %d\n",n);
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

//__device__ void GPUMultHelixPropTranspEndcap(MP6x6F* A, MP6x6F* B, MP6x6SF* C) {
HOSTDEV void MultHelixPropTranspEndcap(MP6x6F* A, MP6x6F* B, MP6x6SF* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
//#pragma omp simd
  //int n = threadIdx.x + blockIdx.x*blockDim.x;
#if USE_ACC
#else
#if USE_GPU
#else
#pragma omp simd
#endif
#endif
  for (int n = 0; n < N; ++n)
//while(n<N)
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
 //  n += blockDim.x*gridDim.x;
  }
}

//__device__ void GPUpropagateToZ(const MP6x6SF* inErr, const MP6F* inPar,const MP1I* inChg,const MP3F* msP, MP6x6SF* outErr, MP6F* outPar) {
HOSTDEV void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar,const MP1I* inChg,const MP3F* msP, MP6x6SF* outErr, MP6F* outPar) {
  //
  MP6x6F errorProp, temp;
  //int it =threadIdx.x;
  //int it =0;
//#pragma omp simd
//while(it<bsize){
#if USE_ACC
#else 
#if USE_GPU
#else
#pragma omp simd
#endif
#endif
for(size_t it=0;it<bsize;++it){
    const float zout = z(msP,it);
    const float k = q(inChg,it)*100/3.8;
    const float deltaZ = zout - z(inPar,it);
    const float pt = 1./ipt(inPar,it);
    const float cosP = cosf(phi(inPar,it));
    const float sinP = sinf(phi(inPar,it));
    const float cosT = cosf(theta(inPar,it));
    const float sinT = sinf(theta(inPar,it));
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float alpha = deltaZ*sinT*ipt(inPar,it)/(cosT*k);
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
    
    for (size_t i=0;i<6;++i) errorProp.data[bsize*GPUPosInMtrx(i,i,6) + it] = 1.;
    errorProp.data[bsize*GPUPosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
    errorProp.data[bsize*GPUPosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*ipt(inPar,it))-k*(cosP*sina-sinP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
    errorProp.data[bsize*GPUPosInMtrx(0,4,6) + it] = (k/ipt(inPar,it))*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp.data[bsize*GPUPosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
    errorProp.data[bsize*GPUPosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
    errorProp.data[bsize*GPUPosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*ipt(inPar,it))-k*(sinP*sina+cosP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
    errorProp.data[bsize*GPUPosInMtrx(1,4,6) + it] = (k/ipt(inPar,it))*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp.data[bsize*GPUPosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
    errorProp.data[bsize*GPUPosInMtrx(4,2,6) + it] = -ipt(inPar,it)*sinT/(cosT*k);
    errorProp.data[bsize*GPUPosInMtrx(4,3,6) + it] = sinT*deltaZ/(cosT*k);
    errorProp.data[bsize*GPUPosInMtrx(4,5,6) + it] = ipt(inPar,it)*deltaZ/(cosT*cosT*k);
 //   it += 1;//blockDim.x;// blockDim.x*gridDim.x;
  }
  
  MultHelixPropEndcap(&errorProp, inErr, &temp);
  MultHelixPropTranspEndcap(&errorProp, &temp, outErr);
  //GPUMultHelixPropEndcap(&errorProp, inErr, &temp);
  //GPUMultHelixPropTranspEndcap(&errorProp, &temp, outErr);
}



__global__ void GPUsequence(ALLTRKS* trk, ALLHITS* hit, ALLTRKS* outtrk, const int stream){
	const int streams = 5;
	for (size_t ie = blockIdx.x; ie<nevts/streams; ie+=gridDim.x){
		for(size_t ib = threadIdx.x; ib <nb; ib+=blockDim.x){
		//for(size_t ie = threadIdx.x; ie <nevts; ie+=blockDim.x){
	//for (size_t ib = blockIdx.x; ib<nb; ib+=gridDim.x){
//			printf("%s %s",ie,ib);
			const MPTRK* btracks = bTk(trk,ie+stream*nevts/streams,ib);
			const MPHIT* bhits = bHit(hit,ie+stream*nevts/streams,ib);
			MPTRK* obtracks = bTk(outtrk,ie+stream*nevts/streams,ib);
			
			propagateToZ(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par);
		}
	}
}

void GPUsequence1(ALLTRKS* trk,ALLHITS* hit, ALLTRKS* outtrk,cudaStream_t* streams){
//void GPUsequence1(ALLTRKS* trk,ALLHITS* hit, ALLTRKS* outtrk){
	//GPUsequence<<<500,600>>>(trk,hit,outtrk);
	GPUsequence<<<200,600,0,streams[0]>>>(trk,hit,outtrk,0);
	GPUsequence<<<200,600,0,streams[1]>>>(trk,hit,outtrk,1);
	GPUsequence<<<200,600,0,streams[2]>>>(trk,hit,outtrk,2);
	GPUsequence<<<200,600,0,streams[3]>>>(trk,hit,outtrk,3);
	GPUsequence<<<200,600,0,streams[4]>>>(trk,hit,outtrk,4);
	//cudaStreamSynchronize(streams[0]);
	//cudaStreamSynchronize(streams[0]);
	cudaDeviceSynchronize();
}

void prefetch(ALLTRKS* trk,ALLHITS* hit, ALLTRKS* outtrk){
	cudaMallocManaged((void**)&outtrk,sizeof(ALLTRKS));
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(trk,sizeof(ALLTRKS),device,NULL);
	cudaMemPrefetchAsync(hit,sizeof(ALLHITS),device,NULL);
	cudaMemPrefetchAsync(outtrk,sizeof(ALLTRKS),device,NULL);
}
