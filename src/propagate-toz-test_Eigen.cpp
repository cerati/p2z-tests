/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3 -I/mnt/data1/dsr/mkfit-hackathon/eigen --expt-relaxed-constexpr -I/mnt/data1/dsr/cub
*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>

#define FIXED_RSEED

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
#define smear 0.1
#include <Eigen/Dense>
#include <Eigen/Core>

#ifndef NITER
#define NITER 5
#endif
#ifndef nlayer
#define nlayer 20
#endif

#ifndef nthreads
#define nthreads 64
#endif

using namespace Eigen;

typedef Matrix<size_t, Dynamic, Dynamic> MatrixXt;
typedef Matrix<size_t, Dynamic, 1> VectorXt;
typedef Matrix<float, Dynamic, 1> VectorXf;

size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

size_t SymOffsets33(size_t i) {
  VectorXt offs(9); 
  offs << 0, 1, 3, 1, 2, 4, 3, 4, 5;
  return offs(i);
}

size_t SymOffsets66(size_t i) {
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

MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

float q(const MP1I* bq, size_t it){
  return (*bq).data[it](0);
}
//
float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it](ipar);
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
  (*bpars).data[it](ipar) = val;
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
const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib, size_t lay) {
  return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it](ipar);
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
  //[DEBUG by Seyong on Dec. 28, 2020] add 4th argument(nlayer-1) to bHit() below.
  const MPHIT* bhits = bHit(hits, ev, ib, nlayer-1);
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
	        result[ib + nb*ie].par.data[it](ip/*bsize*/) = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov
	      for (size_t ip=0;ip<21;++ip) {
	        result[ib + nb*ie].cov.data[it](ip/*bsize*/) = (1+smear*randn(0,1))*inputtrk.cov[ip];
	      }
	      //q
	      result[ib + nb*ie].q.data[it](0) = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      }
    }
  }
  return result;
}

MPHIT* prepareHits(AHIT inputhit) {
  MPHIT* result = (MPHIT*) malloc(nlayer*nevts*nb*sizeof(MPHIT));  //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].pos.data[it](ip/*bsize*/) = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].cov.data[it](ip/*bsize*/) = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return result;
}

#define N bsize
inline void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
  const Matrix<float,6,6> *a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const Matrix<float,6,6> *b = (*B).data; //ASSUME_ALIGNED(b, 64);
  Matrix<float,6,6> *c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#pragma omp simd
  for (int n =0; n< N; ++n){
    c[n]/*.noalias()*/ = a[n]*b[n];
  }
}

inline void MultHelixPropTranspEndcap(const MP6x6F* A, const MP6x6F* B, MP6x6SF* C) {
  const Matrix<float,6,6> *a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const Matrix<float,6,6> *b = (*B).data; //ASSUME_ALIGNED(b, 64);
  Matrix<float,6,6> *c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#pragma omp simd
  for (int n =0; n< N; ++n){
    c[n]/*.noalias()*/= a[n]*b[n].transpose();
  }
}

void KalmanGain(MP6x6SF* A, const MP3x3SF* B, MP6x3* C) {
  // k = P Ht(HPHt + R)^-1
  // HpHt -> cov of x,y,z. take upper 3x3 matrix of P
  // This calculates the inverse of HpHt +R
  const Matrix<float,6,6> *a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const Matrix<float,3,3> *b = (*B).data; //ASSUME_ALIGNED(b, 64);
  Matrix<float,6,3> *c = (*C).data;       //ASSUME_ALIGNED(c, 64);

  Matrix<float,3,3> inter;
  //#pragma omp simd
  for (int n = 0; n < N; ++n)
  {
    inter = ((a[n].block<3,3>(0,0)+b[n]).inverse()) ;
    c[n] = a[n].block<6,3>(0,0) * inter; 
  }
}

void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){
    MP6x3 kGain;
    //KalmanGain(trkErr,hitErr,&kGain);
//#pragma omp simd
  for (size_t it=0;it<bsize;++it) {
    kGain.data[it] = trkErr->data[it].block<6,3>(0,0) * ((trkErr->data[it].block<3,3>(0,0)+hitErr->data[it]).inverse()); //this computes the kalman gain without the function as an intermediary 
    inPar->data[it] = inPar->data[it] + (kGain.data[it]*(msP->data[it]- ((inPar->data[it]).block<3,1>(0,0)))); 
    trkErr->data[it] = trkErr->data[it] - (kGain.data[it]*((trkErr->data[it]).block<3,6>(0,0))); 
  }

}

const float kfact = 100/3.8;
void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar,
		  const MP1I* inChg, const MP3F* msP,
	                MP6x6SF* outErr, MP6F* outPar) {
  //
  MP6x6F errorProp, temp;
#pragma omp simd
  for (size_t it=0;it<bsize;++it) {	
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
    outPar->data[it](0,0) = x(inPar,it) + k*(pxin*sina - pyin*(1.-cosa)) ;
    outPar->data[it](1,0) = y(inPar,it) + k*(pyin*sina + pxin*(1.-cosa)) ;
    outPar->data[it](2,0)= zout;
    outPar->data[it](3,0)= ipt(inPar,it);
    outPar->data[it](4,0)= phi(inPar,it)+alpha ;
    outPar->data[it](5,0) = theta(inPar,it) ;
    
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
    errorProp.data[it](4,5) = ipt(inPar,it)*deltaZ*(icosT*icosTk);
//    errorProp.data[it](0,2) = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
//    errorProp.data[it](0,3) = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*ipt(inPar,it))-k*(cosP*sina-sinP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
//    errorProp.data[it](0,4) = (k/ipt(inPar,it))*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
//    errorProp.data[it](0,5) = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
//    errorProp.data[it](1,2) = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
//    errorProp.data[it](1,3) = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*ipt(inPar,it))-k*(sinP*sina+cosP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
//    errorProp.data[it](1,4) = (k/ipt(inPar,it))*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
//    errorProp.data[it](1,5) = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
//    errorProp.data[it](4,2) = -ipt(inPar,it)*sinT/(cosT*k);
//    errorProp.data[it](4,3) = sinT*deltaZ/(cosT*k);
//    errorProp.data[it](4,5) = ipt(inPar,it)*deltaZ/(cosT*cosT*k);
  }
  MultHelixPropEndcap(&errorProp, inErr, &temp);
  MultHelixPropTranspEndcap(&errorProp, &temp, outErr);
}

int main (int argc, char* argv[]) {
   printf("Running Eigen!\n");
   omp_set_dynamic(0);
   omp_set_num_threads(nthreads);
   Eigen::initParallel();

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
   MPTRK* outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK)); 
   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(int itr=0;itr<NITER;itr++){
#pragma omp parallel for
      for (size_t ie=0;ie<nevts;++ie) { // loop over events
#pragma omp simd
        for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
          const MPTRK* btracks = bTk(trk, ie, ib);
          MPTRK* obtracks = bTk(outtrk, ie, ib);
          for(size_t layer=0;layer<nlayer;++layer){
            const MPHIT* bhits = bHit(hit, ie, ib,layer);
            propagateToZ(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
            KalmanUpdate(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos);
          }
        }
      }
   }

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, nthreads);

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

   free(trk);
   free(hit);
   free(outtrk);

   return 0;
}
