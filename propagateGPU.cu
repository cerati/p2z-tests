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
#include "propagateGPU.cuh"
#include "propagateGPUStructs.cuh"
#include "propagate-toz-test.h"

#define nevts 1000
#define nb    600
#define bsize 16
#define ntrks nb*bsize
#define smear 0.1

__host__ __device__ size_t GPUPosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

__host__ __device__ size_t GPUSymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

__host__ __device__ size_t GPUSymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}

//struct ATRK {
//  float par[6];
//  float cov[21];
//  int q;
//  int hitidx[22];
//};
//
//struct AHIT {
//  float pos[3];
//  float cov[6];
//};
//
//struct MP1I {
//  int data[1*bsize];
//};
//
//struct MP22I {
//  int data[22*bsize];
//};
//
//struct MP3F {
//  float data[3*bsize];
//};
//
//struct MP6F {
//  float data[6*bsize];
//};
//
//struct MP3x3SF {
//  float data[6*bsize];
//};
//
//struct MP6x6SF {
//  float data[21*bsize];
//};
//
//struct MP6x6F {
//  float data[36*bsize];
//};
//
//struct MPTRK {
//  MP6F    par;
//  MP6x6SF cov;
//  MP1I    q;
//  MP22I   hitidx;
//};
//
//struct ALLTRKS {
//  int ismade = 0;
//  MPTRK  btrks[nevts*ntrks];
//};
//
//struct MPHIT {
//  MP3F    pos;
//  MP3x3SF cov;
//};
//
//struct ALLHITS {
//  MPHIT bhits[nevts*ntrks];
//};

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

__host__ __device__ MPTRK* bTk(ALLTRKS* tracks, size_t ev, size_t ib) {
  return &((*tracks).btrks[ib + nb*ev]);
}

__host__ __device__ gMPTRK* GbTk(const ALLTRKS* tracks, size_t ev, size_t ib) {
  return &((*tracks).btrks[ib + nb*ev]);
}
__host__ __device__ const gMPTRK* bTk(const gALLTRKS* tracks, size_t ev, size_t ib) {
  return &((*tracks).btrks[ib + nb*ev]);
}

__host__ __device__ float q(const gMP1I* bq, size_t it){
  return (*bq).data[it];
}
//
__host__ __device__ float par(const gMP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
__host__ __device__ float x    (const gMP6F* bpars, size_t it){ return par(bpars, it, 0); }
__host__ __device__ float y    (const gMP6F* bpars, size_t it){ return par(bpars, it, 1); }
__host__ __device__ float z    (const gMP6F* bpars, size_t it){ return par(bpars, it, 2); }
__host__ __device__ float ipt  (const gMP6F* bpars, size_t it){ return par(bpars, it, 3); }
__host__ __device__ float phi  (const gMP6F* bpars, size_t it){ return par(bpars, it, 4); }
__host__ __device__ float theta(const gMP6F* bpars, size_t it){ return par(bpars, it, 5); }

__host__ __device__ float par(const gMPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
__host__ __device__ float x    (const gMPTRK* btracks, size_t it){ return par(btracks, it, 0); }
__host__ __device__ float y    (const gMPTRK* btracks, size_t it){ return par(btracks, it, 1); }
__host__ __device__ float z    (const gMPTRK* btracks, size_t it){ return par(btracks, it, 2); }
__host__ __device__ float ipt  (const gMPTRK* btracks, size_t it){ return par(btracks, it, 3); }
__host__ __device__ float phi  (const gMPTRK* btracks, size_t it){ return par(btracks, it, 4); }
__host__ __device__ float theta(const gMPTRK* btracks, size_t it){ return par(btracks, it, 5); }
//
__host__ __device__ float par(const gALLTRKS* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const gMPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
__host__ __device__ float x    (const gALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
__host__ __device__ float y    (const gALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
__host__ __device__ float z    (const gALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
__host__ __device__ float ipt  (const gALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
__host__ __device__ float phi  (const gALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
__host__ __device__ float theta(const gALLTRKS* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
__host__ __device__ void setpar(gMP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
__host__ __device__ void setx    (gMP6F* bpars, size_t it, float val){ return setpar(bpars, it, 0, val); }
__host__ __device__ void sety    (gMP6F* bpars, size_t it, float val){ return setpar(bpars, it, 1, val); }
__host__ __device__ void setz    (gMP6F* bpars, size_t it, float val){ return setpar(bpars, it, 2, val); }
__host__ __device__ void setipt  (gMP6F* bpars, size_t it, float val){ return setpar(bpars, it, 3, val); }
__host__ __device__ void setphi  (gMP6F* bpars, size_t it, float val){ return setpar(bpars, it, 4, val); }
__host__ __device__ void settheta(gMP6F* bpars, size_t it, float val){ return setpar(bpars, it, 5, val); }
//
__host__ __device__ void setpar(gMPTRK* btracks, size_t it, size_t ipar, float val){
  return setpar(&(*btracks).par,it,ipar,val);
}
__host__ __device__ void setx    (gMPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 0, val); }
__host__ __device__ void sety    (gMPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 1, val); }
__host__ __device__ void setz    (gMPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 2, val); }
__host__ __device__ void setipt  (gMPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 3, val); }
__host__ __device__ void setphi  (gMPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 4, val); }
__host__ __device__ void settheta(gMPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 5, val); }

__host__ __device__ gMPHIT* GbHit(const ALLHITS* hits, size_t ev, size_t ib) {
  return (*gMPHIT)&((*hits).bhits[ib + nb*ev]);
}
//__host__ __device__ MPHIT* bHit(ALLHITS* hits, size_t ev, size_t ib) {
//  return &((*hits).bhits[ib + nb*ev]);
//}
__host__ __device__ const MPHIT* bHit(const ALLHITS* hits, size_t ev, size_t ib) {
  return &((*hits).bhits[ib + nb*ev]);
}
//
__host__ __device__ float pos(const gMP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
__host__ __device__ float x(const gMP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
__host__ __device__ float y(const gMP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
__host__ __device__ float z(const gMP3F* hpos, size_t it)    { return pos(hpos, it, 2); }
//
__host__ __device__ float pos(const gMPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
__host__ __device__ float x(const gMPHIT* hits, size_t it)    { return pos(hits, it, 0); }
__host__ __device__ float y(const gMPHIT* hits, size_t it)    { return pos(hits, it, 1); }
__host__ __device__ float z(const gMPHIT* hits, size_t it)    { return pos(hits, it, 2); }
//
__host__ __device__ float pos(const ALLHITS* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const gMPHIT* bhits = GbHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
__host__ __device__ float x(const ALLHITS* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
__host__ __device__ float y(const ALLHITS* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
__host__ __device__ float z(const ALLHITS* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }

//ALLTRKS* prepareTracks(ATRK inputtrk) {
//  ALLTRKS* result = (ALLTRKS*) malloc(sizeof(ALLTRKS)); //fixme, align?
//  printf("bool: %d",(*result).ismade);
//  // store in element order for bunches of bsize matrices (a la matriplex)
//  for (size_t ie=0;ie<nevts;++ie) {
//    for (size_t ib=0;ib<nb;++ib) {
//      for (size_t it=0;it<bsize;++it) {
//	//par
//	for (size_t ip=0;ip<6;++ip) {
//	  //printf("randt: %f\n",  (*result).btrks[ib + nb*ie].par.data[it + ip*bsize]);
//	  (*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
//	 // printf("randt: %f\n",  (*result).btrks[ib + nb*ie].par.data[it + ip*bsize]);
//	}
//	//cov
//	for (size_t ip=0;ip<36;++ip) {
//	  (*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
//	}
//	//q
//	(*result).btrks[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
//      }
//    }
//  } 
//  return result;
//}

__global__ 
void GPUprepareTracks(ATRK inputtrk, ALLTRKS* result,const float* trkrandos1,const float* trkrandos2, const float* randoq) {
  
  // store in element order for bunches of bsize matrices (a la matriplex) 
  printf("par: %f,%f,%f,%f,%f,%f\n",inputtrk.par[0],inputtrk.par[1],inputtrk.par[2],inputtrk.par[3],inputtrk.par[4],inputtrk.par[5]);
 
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
	for (size_t ip=0;ip<6;++ip) {
	 // (*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*2)*inputtrk.par[ip];
	  (*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*trkrandos1[ie+ib*nevts+it*nb+ip*bsize])*inputtrk.par[ip];
	  //(*result).btrks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
	}
	//cov
	for (size_t ip=0;ip<36;++ip) {
	  //(*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*2)*inputtrk.cov[ip];
	  (*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*trkrandos2[ie+ib*nevts+it*nb+ip*bsize])*inputtrk.cov[ip];
	  //(*result).btrks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
	}
	//q
	(*result).btrks[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + randoq[0]);//fixme check
      }
    }
  } 
  // printf("results:");// %f\n",(*result).btrks[0].par.data[0]);
  //outtrk = result;
}

__global__ 
void GPUprepareHits(AHIT inputhit, ALLHITS *result,float* hitrandos1,float* hitrandos2) {
  //ALLHITS* result = (ALLHITS*) malloc(sizeof(ALLHITS));  //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
  	//pos
  	for (size_t ip=0;ip<3;++ip) {
  	  //(*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*2)*inputhit.pos[ip];
  	  (*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*hitrandos1[ie+ib*nevts+it*nb+ip*bsize])*inputhit.pos[ip];
  	  //(*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
  	}
  	//cov
  	for (size_t ip=0;ip<6;++ip) {
  	  //(*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*2)*inputhit.cov[ip];
  	  (*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*hitrandos2[ie+ib*nevts+it*nb+ip*bsize])*inputhit.cov[ip];
  	  //(*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
  	}
      }
    }
  }
  //outtrk = result;
}
//ALLHITS* prepareHits(AHIT inputhit) {
//  ALLHITS* result = (ALLHITS*) malloc(sizeof(ALLHITS));  //fixme, align?
//  // store in element order for bunches of bsize matrices (a la matriplex)
//  for (size_t ie=0;ie<nevts;++ie) {
//    for (size_t ib=0;ib<nb;++ib) {
//      for (size_t it=0;it<bsize;++it) {
//  	//pos
//  	for (size_t ip=0;ip<3;++ip) {
//  	  (*result).bhits[ib + nb*ie].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
//  	}
//  	//cov
//  	for (size_t ip=0;ip<6;++ip) {
//  	  (*result).bhits[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
//  	}
//      }
//    }
//  }
//  return result;
//}

#define N bsize
__global__ 
void GPUMultHelixPropEndcap(MP6x6F* A,MP6x6SF* B, MP6x6F* C) {
//__global__ void MultHelixPropEndcap(const float* a, const MP6x6SF* B, float* c) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
//#pragma omp simd
  int n = threadIdx.x + blockIdx.x*blockDim.x;
  //for (int n = 0; n < N; ++n)
while(n<N)
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
   n += blockDim.x*gridDim.x;
  }
}

__global__ void GPUMultHelixPropTranspEndcap(MP6x6F* A, MP6x6F* B, MP6x6SF* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
//#pragma omp simd
  int n = threadIdx.x + blockIdx.x*blockDim.x;
  //for (int n = 0; n < N; ++n)
while(n<N)
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
   n += blockDim.x*gridDim.x;
  }
}

//__device__ MP6x6F errorProp, temp;
//__host__ void GPUpropagateToZ(/*const MP6x6SF* inErr,*/ const MP6F* inPar,
__global__ void GPUpropagateToZ(const gMP6F* inPar,const gMP1I* inChg,const gMP3F* msP, gMP6F* outPar, gMP6x6F errorProp) {
//void GPUpropagateToZ(const MP6F* inPar,const MP1I* inChg,const MP3F* msP, MP6F* outPar, MP6x6F errorProp) {
  //
	//printf("XXX\n");
	//printf("Test1: %f\n",x(inPar,0));	
  //MP6x6F errorProp;//, temp;
  int it =0;
//#pragma omp simd
while(it<bsize){
	printf("YYY\n");
 // printf("this is a test4: %d\n",it);
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
    it += 1;// blockDim.x*gridDim.x;
  }
  //
  //MultHelixPropEndcap(&errorProp, inErr, &temp);
  //MultHelixPropTranspEndcap(&errorProp, &temp, outErr);
  
  //GPUMultHelixPropEndcap<<<256,256>>>(&errorProp, inErr, &temp);
  //cudaDeviceSynchronize();
  //GPUMultHelixPropTranspEndcap<<<256,256>>>(&errorProp, &temp, outErr);
  //cudaDeviceSynchronize();
  //errorProp1 = errorProp; 
}

void allocateManagedx(ALLTRKS* trk_d,ALLHITS* hit_d, ALLTRKS* outtrk_d){
        cudaMallocManaged((void**)&trk_d,sizeof(ALLTRKS));
        cudaMallocManaged((void**)&hit_d,sizeof(ALLHITS));
	cudaMallocManaged((void**)&outtrk_d,sizeof(ALLTRKS));
}
void allocateManaged(gMPTRK* btracks,gMPHIT* bhits, gMPTRK* obtracks){
        cudaMallocManaged((void**)&btracks,sizeof(MPTRK));
        cudaMallocManaged((void**)&bhits,sizeof(MPHIT));
	cudaMallocManaged((void**)&obtracks,sizeof(MPTRK));
}
void allocateGPU(gMPTRK* btracks_d, gMPHIT* bhits_d, gMPTRK* obtracks_d, gMP6x6F errorProp_d, gMP6x6F temp_d){
	cudaMallocManaged((void**)&btracks_d,sizeof(MPTRK));
	cudaMallocManaged((void**)&bhits_d,sizeof(MPHIT));
	cudaMallocManaged((void**)&obtracks_d, sizeof(MPTRK));
	cudaMallocManaged((void**)&errorProp_d, sizeof(MP6x6F));
	cudaMallocManaged((void**)&temp_d, sizeof(MP6x6F));
}
void cpyToGPU(const MPTRK* btracks, MPTRK* btracks_d,const MPHIT* bhits, MPHIT* bhits_d){//, MPTRK* obtracks, MPTRK* obtracks_d){
	//printf("TestX: %f\n",x(btracks,1));	
	//printf("TestY: %f\n",x(btracks_d,1));	
	cudaMemcpy((void*)btracks_d, btracks,sizeof(MPTRK), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)bhits_d, bhits,sizeof(MPHIT), cudaMemcpyHostToDevice);

	cudaMemcpy((void*)&(*btracks_d).par, &(*btracks).par,sizeof(MP6F), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)&(*btracks_d).q, &(*btracks).q,sizeof(MP1I), cudaMemcpyHostToDevice);

	//printf("TestXX: %f\n",x(btracks,1));	
	//printf("TestYY: %f\n",x(btracks_d,1));	
//	cudaMemcpy((void*)obtracks_d, obtracks,sizeof(MPTRK), cudaMemcpyHostToDevice);
}
void cpyFromGPU(MPTRK* obtracks, MPTRK* obtracks_d){
	cudaMemcpy((void*)obtracks, obtracks_d,sizeof(MPTRK), cudaMemcpyDeviceToHost);
}

__global__ void GPUtrackloop(const ALLTRKS* trk, const ALLHITS* hit, ALLTRKS* outtrk, int ie){
//#pragma omp parallel for
	size_t ib = threadIdx.x + blockIdx.x*blockDim.x;
while(ib<nb){	
     //for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
       //
       const MPTRK* btracks = bTk(trk, ie, ib);
       const MPHIT* bhits = bHit(hit, ie, ib);
       MPTRK* obtracks = bTk(outtrk, ie, ib); 
       //GPUpropagateToZ(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
       //propagateToZ<<<1,1>>>(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
	//cudaDeviceSynchronize();
	ib += blockDim.x*gridDim.x;
    }
  }
__global__ void GPUeventloop(const ALLTRKS* trk, const ALLHITS* hit, ALLTRKS* outtrk){
//#pragma omp parallel for
	int ie = threadIdx.x + blockIdx.x*blockDim.x;
   //for (size_t ie=0;ie<nevts;++ie) { // loop over events
	while(ie<nevts){	
	//GPUtrackloop<<<64,32>>>(trk,hit,outtrk,ie); 
	//cudaDeviceSynchronize();
	//__syncthreads();
	ie += blockDim.x*gridDim.x;
	}
}

__global__ void deviceCheck(const MPTRK* btracks_d){
	printf("deviceCheck\n");	
	printf("deviceCheck: %f\n", x(btracks_d,0));	
}

void setValues(const ALLTRKS* trk, const ALLHITS* hit,const ALLTRKS* outtrk, size_t ie,  size_t ib, gMPTRK* btracks, gMPHIT* bhits,gMPTRK* obtracks){
	btracks = GbTk(trk, ie, ib);
        bhits = GbHit(hit, ie, ib);
        obtracks = GbTk(outtrk, ie, ib);
}

//void GPUSequence(const MPTRK* btracks,const MPHIT* bhits,MPTRK* obtracks){
//void GPUSequence(const ALLTRKS* trk, const ALLHITS* hit,const ALLTRKS* outtrk, size_t ie, size_t ib, MPTRK* btracks, MPHIT* bhits,MPTRK* obtracks){
void GPUSequence(const ALLTRKS* trk, const ALLHITS* hit,const ALLTRKS* outtrk, size_t ie, size_t ib){
	cudaSetDevice(0);
	//setValues<<<1,1>>>(trk,hit,outtrk,ie,ib,btracks,bhits,obtracks);
	//MP6x6F errorProp, temp;//	printf("Test1: %f\n",x(obtracks,1));	
	
        //cudaMallocManaged((void**)&errorProp,sizeof(MP6x6F));
        //cudaMallocManaged((void**)&temp,sizeof(MP6x6F));
	

	//printf("running GPUSequence");	
	//printf("Test track: %f\n",x(btracks,1));	
	//printf("Test hit: %f\n",x(bhits,1));	
	//ALLTRKS* outtrk_d;// = (ALLTRKS*) malloc(sizeof(ALLTRKS));
	gMPTRK* btracks_d;// = (MPTRK*)malloc(sizeof(MPTRK));
	gMPHIT* bhits_d;// = (MPHIT*)malloc(sizeof(MPHIT));
	gMPTRK* obtracks_d;// = (MPTRK*)malloc(sizeof(MPTRK));
	gMP6x6F errorProp_d;// = (MP6x6F*)malloc(sizeof(MP6x6F));
	gMP6x6F temp_d;// = (MP6x6F*)malloc(sizeof(MP6x6F));
	allocateGPU(btracks_d,bhits_d,obtracks_d,errorProp_d,temp_d);
	btracks_d = GbTk(trk,ie,ib);
	bhits_d = GbHit(hit,ie,ib);
	obtracks_d = GbTk(outtrk,ie,ib);
	//cpyToGPU(btracks,btracks_d,bhits,bhits_d);
	//allocateManaged(btracks,bhits,obtracks);
	//printf("Test track: %f\n",x(btracks,1));	
	//deviceCheck<<<1,1>>>(btracks);
	///GPUpropagateToZ(&(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).par,errorProp); // vectorized function
	GPUpropagateToZ<<<1,1>>>(&(*btracks_d).par, &(*btracks_d).q, &(*bhits_d).pos, &(*obtracks_d).par,errorProp_d); // vectorized function
	cudaDeviceSynchronize();
//	printf("Test2: %f\n",x(obtracks,1));	
//        GPUMultHelixPropEndcap<<<256,256>>>(&errorProp_d,&(*btracks_d).cov,&temp_d);
//        cudaDeviceSynchronize();
//	printf("Test3: %f\n",x(obtracks,1));	
//        GPUMultHelixPropTranspEndcap<<<256,256>>>(&errorProp_d,&temp_d,&(*obtracks_d).cov);
//	cudaDeviceSynchronize();
//	cpyFromGPU(obtracks,obtracks_d);
//	printf("Test4: %f\n",x(obtracks,1));	


//cudaFree(outtrk_d);
//cudaFree(bhits_d);
//cudaFree(btracks_d);
//cudaFree(obtracks_d);
//cudaFree(&errorProp_d);
//cudaFree(&temp_d);
}

