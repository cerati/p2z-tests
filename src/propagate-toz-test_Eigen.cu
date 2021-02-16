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
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <iomanip>

#define FIXED_RSEED

#ifndef bsize
#define bsize 1
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    (ntrks/bsize)
#ifndef nevts
#define nevts 100
#endif
#define smear 0.1

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
#define threadsperblockx 32
#endif
#define threadsperblocky 512/threadsperblockx
#ifndef blockspergrid
#define blockspergrid 10
#endif

#define HOSTDEV __host__ __device__

using namespace Eigen;
//using Eigen::VectorXt;
//typedef Matrix<size_t, Dynamic, Dynamic> MatrixXt;
//typedef Matrix<size_t, Dynamic, 1> VectorXt;
//typedef Matrix<float, Dynamic, 1> VectorXf;


HOSTDEV size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

HOSTDEV size_t SymOffsets33(size_t i) {
  VectorXd offs(9);
  offs << 0, 1, 3, 1, 2, 4, 3, 4, 5;
  return offs(i);
}

HOSTDEV size_t SymOffsets66(size_t i) {
  VectorXf offs(36);
  offs << 0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20;
  return offs(i);
}

struct ATRK {
  Matrix<float,6,1> par;
  Matrix<float,21,1> cov;
  int q;
//  Matrix<float,22,1> hitidx;
};

struct AHIT {
  Matrix<float,3,1> pos;
  Matrix<float,6,1> cov;
};

struct MP1I {
  Matrix<int,1,1> data[bsize];
};
struct MP22I {
  Matrix<int,22,1> data[bsize];
};

struct MP3F {
  Vector3f data[bsize];
};

struct MP6F {
  Matrix<float,6,1> data[bsize];
};

struct MP3x3 {
  Matrix<float,3,3> data[bsize];
};
struct MP6x3 {
  Matrix<float,6,3> data[bsize];
};

struct MP3x3SF {
  Matrix3f data[bsize];
};

struct MP6x6SF {
  Matrix<float,6,6> data[bsize];
};

struct MP6x6F {
  Matrix<float,6,6> data[bsize];
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
  //MPTRK* result = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
  MPTRK* result;
  cudaMallocHost((void**)&result,nevts*nb*sizeof(MPTRK));
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //par
        for (size_t ip=0;ip<6;++ip) {
          result[ib + nb*ie].par.data[it](ip) = (1+smear*randn(0,1))*inputtrk.par[ip];
        }
        //cov
        for (size_t ip=0;ip<21;++ip) {
          result[ib + nb*ie].cov.data[it](ip) = (1+smear*randn(0,1))*inputtrk.cov[ip];
        }
        //q
        result[ib + nb*ie].q.data[it](0) = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      }
    }
  }
  return result;
}

MPHIT* prepareHits(AHIT inputhit) {
  //MPHIT* result = (MPHIT*) malloc(nlayer*nevts*nb*sizeof(MPHIT));
  MPHIT* result;
  cudaMallocHost((void**)&result,nlayer*nevts*nb*sizeof(MPHIT));
  for (int lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
          //pos
          for (size_t ip=0;ip<3;++ip) {
            result[lay+nlayer*(ib + nb*ie)].pos.data[it](ip) = (1+smear*randn(0,1))*inputhit.pos[ip];
          }
          //cov
          for (size_t ip=0;ip<6;++ip) {
            result[lay+nlayer*(ib + nb*ie)].cov.data[it](ip) = (1+smear*randn(0,1))*inputhit.cov[ip];
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


HOSTDEV float q(const MP1I* bq, size_t it){
  return (*bq).data[it](0);
}

HOSTDEV float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it](ipar);
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
  (*bpars).data[it](ipar) = val;
}
HOSTDEV void setx    (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 0, val); }
HOSTDEV void sety    (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 1, val); }
HOSTDEV void setz    (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 2, val); }
HOSTDEV void setipt  (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 3, val); }
HOSTDEV void setphi  (MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 4, val); }
HOSTDEV void settheta(MP6F* bpars, size_t it, float val){ return setpar(bpars, it, 5, val); }

HOSTDEV void setpar(MPTRK* btracks, size_t it, size_t ipar, float val){
  return setpar(&(*btracks).par,it,ipar,val);
}
HOSTDEV void setx    (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 0, val); }
HOSTDEV void sety    (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 1, val); }
HOSTDEV void setz    (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 2, val); }
HOSTDEV void setipt  (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 3, val); }
HOSTDEV void setphi  (MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 4, val); }
HOSTDEV void settheta(MPTRK* btracks, size_t it, float val){ return setpar(btracks, it, 5, val); }

HOSTDEV MPHIT* bHit(MPHIT* hits, size_t ev, size_t ib, int lay) {
  return &(hits[lay+nlayer*(ib + nb*ev)]);
}
HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib,int lay) {
  return &(hits[lay+nlayer*(ib + nb*ev)]);
}
HOSTDEV MPHIT* bHit(MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}

HOSTDEV float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it](ipar);
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
  const Matrix<float,6,6> *a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const Matrix<float,6,6> *b = (*B).data; //ASSUME_ALIGNED(b, 64);
  Matrix<float,6,6> *c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x){
    c[it]/*.noalias()*/ = a[it]*b[it];
  }
}

__forceinline__ __device__ void MultHelixPropTranspEndcap(MP6x6F* A, MP6x6F* B, MP6x6SF* C) {
  const Matrix<float,6,6> *a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const Matrix<float,6,6> *b = (*B).data; //ASSUME_ALIGNED(b, 64);
  Matrix<float,6,6> *c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x){
    c[it].noalias()= a[it]*b[it].transpose();
  }
}

__device__ __forceinline__ void KalmanGain(MP6x6SF* A, const MP3x3SF* B, MP6x3* C) {
  // k = P Ht(HPHt + R)^-1
  // HpHt -> cov of x,y,z. take upper 3x3 matrix of P
  // This calculates the inverse of HpHt +R
  const Matrix<float,6,6> *a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const Matrix<float,3,3> *b = (*B).data; //ASSUME_ALIGNED(b, 64);
  Matrix<float,6,3> *c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  Matrix<float,3,3> inter;
  for (int n = 0; n < N; ++n)
  {
    inter = ((a[n].block<3,3>(0,0)+b[n]).inverse()) ;
    c[n] = a[n].block<6,3>(0,0) * inter;
  }
}

__device__ __forceinline__ void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){
    MP6x3 kGain;
    //KalmanGain(trkErr,hitErr,&kGain);
//#pragma omp simd
  for (size_t it=0;it<bsize;++it) {
    kGain.data[it] = trkErr->data[it].block<6,3>(0,0) * ((trkErr->data[it].block<3,3>(0,0)+hitErr->data[it]).inverse()); // computes the full kalman gain without the funtion
    //if(isnan(kGain.data[it].sum())) {
    //Matrix<float,6,1> zeros;
    //zeros << 0,0,0,0,0,0;
    //inPar->data[it] = zeros;
    //continue;}
    //inPar->data[it] = test;//inPar->data[it] + (kGain.data[it]*(msP->data[it]- ((inPar->data[it]).block<3,1>(0,0))));
   // if(it==0){
   // std::cout<<(kGain.data[it]*(msP->data[it]- ((inPar->data[it]).block<3,1>(0,0))))<<"done"<<std::endl;
   // }
    inPar->data[it] = inPar->data[it] + (kGain.data[it]*(msP->data[it]- ((inPar->data[it]).block<3,1>(0,0))));
    trkErr->data[it] = trkErr->data[it] - (kGain.data[it]*((trkErr->data[it]).block<3,6>(0,0)));
  }

}

__device__ __constant__ float kfact = 100/3.8;
__device__ __forceinline__ void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar,
			  const MP1I* inChg,const MP3F* msP, MP6x6SF* outErr, MP6F* outPar) {
  MP6x6F errorProp, temp;
  for(size_t it=threadIdx.x;it<bsize;it+=blockDim.x){
    const float zout = z(msP,it);
    const float k = q(inChg,it)*kfact;//100/3.8;
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
    outPar->data[it](0,0) = x(inPar,it) + k*(pxin*sina - pyin*(1.-cosa));
    outPar->data[it](1,0) = y(inPar,it) + k*(pyin*sina + pxin*(1.-cosa));
    outPar->data[it](2,0) = zout;
    outPar->data[it](3,0) = ipt(inPar,it);
    outPar->data[it](4,0) = phi(inPar,it)+alpha;
    outPar->data[it](5,0) = theta(inPar,it);
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
    
    for (size_t i=0;i<6;++i) errorProp.data[it](i,i) = 1.;
    errorProp.data[it](0,2) = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp.data[it](0,3) = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.-cCosPsina))*(pt*pt);
    errorProp.data[it](0,4) = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp.data[it](0,5) = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*icosT);
    errorProp.data[it](1,2) = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp.data[it](1,3) = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.-cCosPsina))*(pt*pt);
    errorProp.data[it](1,4) = (k*pt)*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp.data[it](1,5) = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp.data[it](4,2) = -ipt(inPar,it)*sinT*(icosTk);
    errorProp.data[it](4,3) = sinT*deltaZ*(icosTk);
    errorProp.data[it](4,5) = ipt(inPar,it)*deltaZ*icosT*icosTk;
    //errorProp.data[it](4,5) = ipt(inPar,it)*deltaZ*(icosT*icosTk);

   // errorProp.data[it](0,2) = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
   // errorProp.data[it](0,3) = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*ipt(inPar,it))-k*(cosP*sina-sinP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
   // errorProp.data[it](0,4) = (k/ipt(inPar,it))*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
   // errorProp.data[it](0,5) = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
   // errorProp.data[it](1,2) = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
   // errorProp.data[it](1,3) = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*ipt(inPar,it))-k*(sinP*sina+cosP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
   // errorProp.data[it](1,4) = (k/ipt(inPar,it))*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
   // errorProp.data[it](1,5) = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
   // errorProp.data[it](4,2) = -ipt(inPar,it)*sinT/(cosT*k);
   // errorProp.data[it](4,3) = sinT*deltaZ/(cosT*k);
    //errorProp.data[it](4,5) = ipt(inPar,it)*deltaZ/(cosT*cosT*k);

  }
  __syncthreads();
  MultHelixPropEndcap(&errorProp, inErr, &temp);
  __syncthreads();
  MultHelixPropTranspEndcap(&errorProp, &temp, outErr);
}



__global__ void GPUsequence(MPTRK* trk, MPHIT* hit, MPTRK* outtrk, const int stream){
  int ie_range;
  if(stream == num_streams){ie_range = (int)(nevts%num_streams);}
  else{ie_range= (int)(nevts/num_streams);}
  for (size_t ie = blockIdx.x; ie<ie_range; ie+=gridDim.x){
    for(size_t ib = threadIdx.y; ib <nb; ib+=blockDim.y){
      const MPTRK* btracks = bTk(trk,ie,ib);
      MPTRK* obtracks = bTk(outtrk,ie,ib);
      for(int layer=0; layer<nlayer;++layer){
        const MPHIT* bhits = bHit(hit,ie,ib,layer);	
        propagateToZ(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, 
                     &(*obtracks).cov, &(*obtracks).par);
        __syncthreads();
        KalmanUpdate(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos);
      }
    }
  }
}


void transfer(MPTRK* trk, MPHIT* hit, MPTRK* trk_dev, MPHIT* hit_dev){

   
  cudaMemcpy(trk_dev, trk, nevts*nb*sizeof(MPTRK), cudaMemcpyHostToDevice);
  cudaMemcpy(&trk_dev->par, &trk->par, sizeof(MP6F), cudaMemcpyHostToDevice);
  cudaMemcpy(&((trk_dev->par).data), &((trk->par).data), bsize*sizeof(Matrix<float,6,1>), cudaMemcpyHostToDevice);
  cudaMemcpy(&trk_dev->cov, &trk->cov, sizeof(MP6x6SF), cudaMemcpyHostToDevice);
  cudaMemcpy(&((trk_dev->cov).data), &((trk->cov).data), bsize*sizeof(Matrix<float,6,6>), cudaMemcpyHostToDevice);
  cudaMemcpy(&trk_dev->q, &trk->q, sizeof(MP1I), cudaMemcpyHostToDevice);
  cudaMemcpy(&((trk_dev->q).data), &((trk->q).data), bsize*sizeof(Matrix<int,1,1>), cudaMemcpyHostToDevice);

  cudaMemcpy(hit_dev,hit,nlayer*nevts*nb*sizeof(MPHIT), cudaMemcpyHostToDevice);
  cudaMemcpy(&hit_dev->pos,&hit->pos,sizeof(MP3F), cudaMemcpyHostToDevice);
  cudaMemcpy(&(hit_dev->pos).data,&(hit->pos).data,bsize*sizeof(Matrix<float,3,1>), cudaMemcpyHostToDevice);
  cudaMemcpy(&hit_dev->cov,&hit->cov,sizeof(MP3x3SF), cudaMemcpyHostToDevice);
  cudaMemcpy(&(hit_dev->cov).data,&(hit->cov).data,bsize*sizeof(Matrix<float,3,3>), cudaMemcpyHostToDevice);
}
void transfer_back(MPTRK* trk, MPTRK* trk_host){
  cudaMemcpy(trk_host, trk, nevts*nb*sizeof(MPTRK), cudaMemcpyDeviceToHost);
  cudaMemcpy(&trk_host->par, &trk->par, sizeof(MP6F), cudaMemcpyDeviceToHost);
  cudaMemcpy(&((trk_host->par).data), &((trk->par).data), bsize*sizeof(Matrix<float,6,1>), cudaMemcpyDeviceToHost);
  cudaMemcpy(&trk_host->cov, &trk->cov, sizeof(MP6x6SF), cudaMemcpyDeviceToHost);
  cudaMemcpy(&((trk_host->cov).data), &((trk->cov).data), bsize*sizeof(Matrix<float,6,6>), cudaMemcpyDeviceToHost);
  cudaMemcpy(&trk_host->q, &trk->q, sizeof(MP1I), cudaMemcpyDeviceToHost);
  cudaMemcpy(&((trk_host->q).data), &((trk->q).data), bsize*sizeof(Matrix<int,1,1>), cudaMemcpyDeviceToHost);
}



int main (int argc, char* argv[]) {

  printf("RUNNING CUDA!!\n");
  ATRK inputtrk = {
     {-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975},
     {6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
      6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,
      0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348},
     1
  };

  AHIT inputhit = {
     {-20.7824649810791, -12.24150276184082, 57.8067626953125},
     {2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945}
  };
  printf("track in pos: %f, %f, %f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2]);
  printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66(PosInMtrx(0,0,6))],
                                              inputtrk.cov[SymOffsets66(PosInMtrx(1,1,6))],
                                              inputtrk.cov[SymOffsets66(PosInMtrx(2,2,6))]);
  printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);

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
  MPTRK* trk = prepareTracks(inputtrk);
  MPHIT* hit = prepareHits(inputhit);
  MPTRK* trk_dev;
  MPHIT* hit_dev;
  MPTRK* outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
  MPTRK* outtrk_dev;
  cudaMalloc((MPTRK**)&trk_dev,nevts*nb*sizeof(MPTRK));  
  cudaMalloc((MPTRK**)&hit_dev,nlayer*nevts*nb*sizeof(MPHIT));
  cudaMalloc((MPTRK**)&outtrk_dev,nevts*nb*sizeof(MPTRK));  

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
    cudaStreamCreateWithFlags(&streams[s],cudaStreamNonBlocking);
  }

  gettimeofday(&timecheck, NULL);
  setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
 

  printf("done preparing!\n");
  printf("dev: %f\n",trk->par.data[0](0));
  
  printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
  printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
  printf("Size of struct struct MPHIT hit[] = %ld\n", nlayer*nevts*nb*sizeof(struct MPHIT));

  auto wall_start = std::chrono::high_resolution_clock::now();

  for(int itr=0; itr<NITER; itr++){
    for (int s = 0; s<num_streams;s++){
      cudaMemcpyAsync(trk_dev+(s*stream_chunk), trk+(s*stream_chunk), stream_chunk*sizeof(MPTRK), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(trk_dev+(s*stream_chunk))->par, &(trk+(s*stream_chunk))->par, sizeof(MP6F), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(((trk_dev+(s*stream_chunk))->par).data), &(((trk+(s*stream_chunk))->par).data), 6*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(trk_dev+(s*stream_chunk))->cov, &(trk+(s*stream_chunk))->cov, sizeof(MP6x6SF), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(((trk_dev+(s*stream_chunk))->cov).data), &(((trk+(s*stream_chunk))->cov).data), 36*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(trk_dev+(s*stream_chunk))->q, &(trk+(s*stream_chunk))->q, sizeof(MP1I), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(((trk_dev+(s*stream_chunk))->q).data), &(((trk+(s*stream_chunk))->q).data), 1*bsize*sizeof(int), cudaMemcpyHostToDevice, streams[s]);
      
      cudaMemcpyAsync(hit_dev+(s*stream_chunk*nlayer),hit+(s*stream_chunk),nlayer*stream_chunk*sizeof(MPHIT), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(hit_dev+(s*stream_chunk*nlayer))->pos,&(hit+(s*stream_chunk*nlayer))->pos,sizeof(MP3F), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&((hit_dev+(s*stream_chunk*nlayer))->pos).data,&((hit+(s*stream_chunk*nlayer))->pos).data,3*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&(hit_dev+(s*stream_chunk*nlayer))->cov,&(hit+(s*stream_chunk*nlayer))->cov,sizeof(MP3x3SF), cudaMemcpyHostToDevice, streams[s]);
      cudaMemcpyAsync(&((hit_dev+(s*stream_chunk*nlayer))->cov).data,&((hit+(s*stream_chunk*nlayer))->cov).data,6*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
    }
    if(stream_remainder != 0){
      cudaMemcpyAsync(trk_dev+(num_streams*stream_chunk), trk+(num_streams*stream_chunk), stream_remainder*sizeof(MPTRK), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(trk_dev+(num_streams*stream_chunk))->par, &(trk+(num_streams*stream_chunk))->par, sizeof(MP6F), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(((trk_dev+(num_streams*stream_chunk))->par).data), &(((trk+(num_streams*stream_chunk))->par).data), 6*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(trk_dev+(num_streams*stream_chunk))->cov, &(trk+(num_streams*stream_chunk))->cov, sizeof(MP6x6SF), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(((trk_dev+(num_streams*stream_chunk))->cov).data), &(((trk+(num_streams*stream_chunk))->cov).data), 36*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(trk_dev+(num_streams*stream_chunk))->q, &(trk+(num_streams*stream_chunk))->q, sizeof(MP1I), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(((trk_dev+(num_streams*stream_chunk))->q).data), &(((trk+(num_streams*stream_chunk))->q).data), 1*bsize*sizeof(int), cudaMemcpyHostToDevice, streams[num_streams]);
      
      cudaMemcpyAsync(hit_dev+(num_streams*stream_chunk*nlayer),hit+(num_streams*stream_chunk*nlayer),nlayer*stream_remainder*sizeof(MPHIT), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(hit_dev+(num_streams*stream_chunk*nlayer))->pos,&(hit+(num_streams*stream_chunk*nlayer))->pos,sizeof(MP3F), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&((hit_dev+(num_streams*stream_chunk*nlayer))->pos).data,&((hit+(num_streams*stream_chunk*nlayer))->pos).data,3*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&(hit_dev+(num_streams*stream_chunk*nlayer))->cov,&(hit+(num_streams*stream_chunk*nlayer))->cov,sizeof(MP3x3SF), cudaMemcpyHostToDevice, streams[num_streams]);
      cudaMemcpyAsync(&((hit_dev+(num_streams*stream_chunk*nlayer))->cov).data,&((hit+(num_streams*stream_chunk*nlayer))->cov).data,6*bsize*sizeof(float), cudaMemcpyHostToDevice, streams[num_streams]);
    }

    for (int s = 0; s<num_streams;s++){
      GPUsequence<<<grid,block,0,streams[s]>>>(trk_dev+(s*stream_chunk),hit_dev+(s*stream_chunk*nlayer),outtrk_dev+(s*stream_chunk),s);
    }  
    if(stream_remainder != 0){
      GPUsequence<<<grid,block,0,streams[num_streams]>>>(trk_dev+(num_streams*stream_chunk),hit_dev+(num_streams*stream_chunk*nlayer),outtrk_dev+(num_streams*stream_chunk),num_streams); 
    }
  
    for (int s = 0; s<num_streams;s++){
      cudaMemcpyAsync(outtrk+(s*stream_chunk), outtrk_dev+(s*stream_chunk), stream_chunk*sizeof(MPTRK), cudaMemcpyDeviceToHost, streams[s]);
      cudaMemcpyAsync(&(outtrk+(s*stream_chunk))->par, &(outtrk_dev+(s*stream_chunk))->par, sizeof(MP6F), cudaMemcpyDeviceToHost, streams[s]);
      cudaMemcpyAsync(&(((outtrk+(s*stream_chunk))->par).data), &(((outtrk_dev+(s*stream_chunk))->par).data), 6*bsize*sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
      cudaMemcpyAsync(&(outtrk+(s*stream_chunk))->cov, &(outtrk_dev+(s*stream_chunk))->cov, sizeof(MP6x6SF), cudaMemcpyDeviceToHost, streams[s]);
      cudaMemcpyAsync(&(((outtrk+(s*stream_chunk))->cov).data), &(((outtrk_dev+(s*stream_chunk))->cov).data), 36*bsize*sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
      cudaMemcpyAsync(&(outtrk+(s*stream_chunk))->q, &(outtrk_dev+(s*stream_chunk))->q, sizeof(MP1I), cudaMemcpyDeviceToHost, streams[s]);
      cudaMemcpyAsync(&(((outtrk+(s*stream_chunk))->q).data), &(((outtrk_dev+(s*stream_chunk))->q).data), 1*bsize*sizeof(int), cudaMemcpyDeviceToHost, streams[s]);
    }
    if(stream_remainder != 0){
      cudaMemcpyAsync(outtrk+(num_streams*stream_chunk), outtrk_dev+(num_streams*stream_chunk), stream_remainder*sizeof(MPTRK), cudaMemcpyDeviceToHost, streams[num_streams]);
      cudaMemcpyAsync(&(outtrk+(num_streams*stream_chunk))->par, &(outtrk_dev+(num_streams*stream_chunk))->par, sizeof(MP6F), cudaMemcpyDeviceToHost, streams[num_streams]);
      cudaMemcpyAsync(&(((outtrk+(num_streams*stream_chunk))->par).data), &(((outtrk_dev+(num_streams*stream_chunk))->par).data), 6*bsize*sizeof(float), cudaMemcpyDeviceToHost, streams[num_streams]);
      cudaMemcpyAsync(&(outtrk+(num_streams*stream_chunk))->cov, &(outtrk_dev+(num_streams*stream_chunk))->cov, sizeof(MP6x6SF), cudaMemcpyDeviceToHost, streams[num_streams]);
      cudaMemcpyAsync(&(((outtrk+(num_streams*stream_chunk))->cov).data), &(((outtrk_dev+(num_streams*stream_chunk))->cov).data), 36*bsize*sizeof(float), cudaMemcpyDeviceToHost, streams[num_streams]);
      cudaMemcpyAsync(&(outtrk+(num_streams*stream_chunk))->q, &(outtrk_dev+(num_streams*stream_chunk))->q, sizeof(MP1I), cudaMemcpyDeviceToHost, streams[num_streams]);
      cudaMemcpyAsync(&(((outtrk+(num_streams*stream_chunk))->q).data), &(((outtrk_dev+(num_streams*stream_chunk))->q).data), 1*bsize*sizeof(int), cudaMemcpyDeviceToHost, streams[num_streams]);
    }
  } //end itr loop

  cudaDeviceSynchronize(); // shaves a few seconds
  auto wall_stop = std::chrono::high_resolution_clock::now();

  
  for (int s = 0; s<stream_range;s++){
    cudaStreamDestroy(streams[s]);
  }
 
   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, num_streams);

   int bad_count =0;
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
       if( isnan(x_) && isnan(y_) && isnan(z_)){bad_count++;continue;} // counts and skips over bad values
       avgpt += pt_;
       avgphi += phi_;
       avgtheta += theta_;
       avgx += x_;
       avgy += y_;
       avgz += z_;
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
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
       if( isnan(x_) && isnan(y_) && isnan(z_)){continue;}
       stdx += (x_-avgx)*(x_-avgx);
       stdy += (y_-avgy)*(y_-avgy);
       stdz += (z_-avgz)*(z_-avgz);
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
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
   printf("bad track evaluation=%d/%d (%f%%)\n", bad_count,nevts*ntrks,100*(float)bad_count/(nevts*ntrks));
	
   cudaFree(trk);
   cudaFree(hit);
   cudaFree(outtrk);
   
return 0;
}

