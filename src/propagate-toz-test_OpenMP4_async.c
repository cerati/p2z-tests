/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

//#define DUMP_OUTPUT
#define FIXED_RSEED

#ifndef bsize
#define bsize 32
#ifdef _OPENARC_
#pragma openarc #define bsize 32
#endif
#endif

#ifndef ntrks
#define ntrks 9600
#ifdef _OPENARC_
#pragma openarc #define ntrks 9600
#endif
#endif

#define nb    (ntrks/bsize)
#ifdef _OPENARC_
#pragma openarc #define nb    (ntrks/bsize)
#endif

#ifndef nevts
#define nevts 100
#ifdef _OPENARC_
#pragma openarc #define nevts 100
#endif
#endif

#define smear 0.00001

#ifndef NITER
#define NITER 5
#endif

#ifndef nlayer
#define nlayer 20
#ifdef _OPENARC_
#pragma openarc #define nlayer 20
#endif
#endif

#ifndef num_streams
#define num_streams 10
#ifdef _OPENARC_
#pragma openarc #define num_streams 10
#endif
#endif


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
  struct MP6F    par;
  struct MP6x6SF cov;
  struct MP1I    q;
  //  struct MP22I   hitidx;
};

struct MPHIT {
  struct MP3F    pos;
  struct MP3x3SF cov;
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

struct MPTRK* bTk(struct MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

const struct MPTRK* bTkC(const struct MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

int q(const struct MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
float par1(const struct MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
float x1    (const struct MP6F* bpars, size_t it){ return par1(bpars, it, 0); }
float _y1    (const struct MP6F* bpars, size_t it){ return par1(bpars, it, 1); }
float z1    (const struct MP6F* bpars, size_t it){ return par1(bpars, it, 2); }
float ipt1  (const struct MP6F* bpars, size_t it){ return par1(bpars, it, 3); }
float phi1  (const struct MP6F* bpars, size_t it){ return par1(bpars, it, 4); }
float theta1(const struct MP6F* bpars, size_t it){ return par1(bpars, it, 5); }
//
float par2(const struct MPTRK* btracks, size_t it, size_t ipar){
  return par1(&(*btracks).par,it,ipar);
}
float x2    (const struct MPTRK* btracks, size_t it){ return par2(btracks, it, 0); }
float y2    (const struct MPTRK* btracks, size_t it){ return par2(btracks, it, 1); }
float z2    (const struct MPTRK* btracks, size_t it){ return par2(btracks, it, 2); }
float ipt2  (const struct MPTRK* btracks, size_t it){ return par2(btracks, it, 3); }
float phi2  (const struct MPTRK* btracks, size_t it){ return par2(btracks, it, 4); }
float theta2(const struct MPTRK* btracks, size_t it){ return par2(btracks, it, 5); }
//
float par3(const struct MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const struct MPTRK* btracks = bTkC(tracks, ev, ib);
  size_t it = tk % bsize;
  return par2(btracks, it, ipar);
}
float x3    (const struct MPTRK* tracks, size_t ev, size_t tk){ return par3(tracks, ev, tk, 0); }
float y3    (const struct MPTRK* tracks, size_t ev, size_t tk){ return par3(tracks, ev, tk, 1); }
float z3    (const struct MPTRK* tracks, size_t ev, size_t tk){ return par3(tracks, ev, tk, 2); }
float ipt3  (const struct MPTRK* tracks, size_t ev, size_t tk){ return par3(tracks, ev, tk, 3); }
float phi3  (const struct MPTRK* tracks, size_t ev, size_t tk){ return par3(tracks, ev, tk, 4); }
float theta3(const struct MPTRK* tracks, size_t ev, size_t tk){ return par3(tracks, ev, tk, 5); }
//
void setpar1(struct MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
void setx1    (struct MP6F* bpars, size_t it, float val){ setpar1(bpars, it, 0, val); }
void sety1    (struct MP6F* bpars, size_t it, float val){ setpar1(bpars, it, 1, val); }
void setz1    (struct MP6F* bpars, size_t it, float val){ setpar1(bpars, it, 2, val); }
void setipt1  (struct MP6F* bpars, size_t it, float val){ setpar1(bpars, it, 3, val); }
void setphi1  (struct MP6F* bpars, size_t it, float val){ setpar1(bpars, it, 4, val); }
void settheta1(struct MP6F* bpars, size_t it, float val){ setpar1(bpars, it, 5, val); }
//
void setpar2(struct MPTRK* btracks, size_t it, size_t ipar, float val){
  setpar1(&(*btracks).par,it,ipar,val);
}
void setx2    (struct MPTRK* btracks, size_t it, float val){ setpar2(btracks, it, 0, val); }
void sety2    (struct MPTRK* btracks, size_t it, float val){ setpar2(btracks, it, 1, val); }
void setz2    (struct MPTRK* btracks, size_t it, float val){ setpar2(btracks, it, 2, val); }
void setipt2  (struct MPTRK* btracks, size_t it, float val){ setpar2(btracks, it, 3, val); }
void setphi2  (struct MPTRK* btracks, size_t it, float val){ setpar2(btracks, it, 4, val); }
void settheta2(struct MPTRK* btracks, size_t it, float val){ setpar2(btracks, it, 5, val); }

const struct MPHIT* bHit(const struct MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
const struct MPHIT* bHit4(const struct MPHIT* hits, size_t ev, size_t ib,size_t lay) {
  return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float pos1(const struct MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x_pos1(const struct MP3F* hpos, size_t it)    { return pos1(hpos, it, 0); }
float y_pos1(const struct MP3F* hpos, size_t it)    { return pos1(hpos, it, 1); }
float z_pos1(const struct MP3F* hpos, size_t it)    { return pos1(hpos, it, 2); }
//
float pos2(const struct MPHIT* hits, size_t it, size_t ipar){
  return pos1(&(*hits).pos,it,ipar);
}
float x_pos2(const struct MPHIT* hits, size_t it)    { return pos2(hits, it, 0); }
float y_pos2(const struct MPHIT* hits, size_t it)    { return pos2(hits, it, 1); }
float z_pos2(const struct MPHIT* hits, size_t it)    { return pos2(hits, it, 2); }
//
float pos3(const struct MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  //[DEBUG on Dec. 22, 2020] change bHit() to bHit4()
  //const struct MPHIT* bhits = bHit(hits, ev, ib);
  const struct MPHIT* bhits = bHit4(hits, ev, ib, nlayer-1);
  size_t it = tk % bsize;
  return pos2(bhits,it,ipar);
}
float x_pos3(const struct MPHIT* hits, size_t ev, size_t tk)    { return pos3(hits, ev, tk, 0); }
float y_pos3(const struct MPHIT* hits, size_t ev, size_t tk)    { return pos3(hits, ev, tk, 1); }
float z_pos3(const struct MPHIT* hits, size_t ev, size_t tk)    { return pos3(hits, ev, tk, 2); }

struct MPTRK* prepareTracks(struct ATRK inputtrk) {
  struct MPTRK* result = (struct MPTRK*) malloc(nevts*nb*sizeof(struct MPTRK)); //fixme, align?
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

struct MPHIT* prepareHits(struct AHIT* inputhits) {
  struct MPHIT* result = (struct MPHIT*) malloc(nlayer*nevts*nb*sizeof(struct MPHIT));  //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
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

#define N bsize
#ifdef _OPENARC_
#pragma openarc #define N bsize
#endif
#pragma omp declare target
void MultHelixPropEndcap(const struct MP6x6F* A, const struct MP6x6SF* B, struct MP6x6F* C) {
 #pragma omp parallel for num_threads(N)
  for (int n = 0; n < N; ++n)
  {
    const float* a = A->data; //ASSUME_ALIGNED(a, 64);
    const float* b = B->data; //ASSUME_ALIGNED(b, 64);
    float* c = C->data;       //ASSUME_ALIGNED(c, 64);
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
#pragma omp end declare target

#pragma omp declare target
void MultHelixPropTranspEndcap(const struct MP6x6F* A, const struct MP6x6F* B, struct MP6x6SF* C) {
 #pragma omp parallel for num_threads(N)
  for (int n = 0; n < N; ++n)
  {
    const float* a = A->data; //ASSUME_ALIGNED(a, 64);
    const float* b = B->data; //ASSUME_ALIGNED(b, 64);
    float* c = C->data;       //ASSUME_ALIGNED(c, 64);
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
#pragma omp end declare target

#pragma omp declare target
void KalmanGainInv(const struct MP6x6SF* A, const struct MP3x3SF* B, struct MP3x3* C) {
 #pragma omp parallel for num_threads(N)
  for (int n = 0; n < N; ++n)
  {
    const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
    const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
    float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
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
#pragma omp end declare target

#pragma omp declare target
void KalmanGain(const struct MP6x6SF* A, const struct MP3x3* B, struct MP3x6* C) {
 #pragma omp parallel for num_threads(N)
  for (int n = 0; n < N; ++n)
  {
    const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
    const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
    float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
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
#pragma omp end declare target

#pragma omp declare target
void KalmanUpdate(struct MP6x6SF* trkErr, struct MP6F* inPar, const struct MP3x3SF* hitErr, const struct MP3F* msP, struct MP3x3* inverse_temp, struct MP3x6* kGain, struct MP6x6SF* newErr){
  //struct MP3x3 inverse_temp1;
  //struct MP3x6 kGain1;
  //struct MP6x6SF newErr1;
  //inverse_temp = &inverse_temp1;
  //kGain = &kGain1;
  //newErr = &newErr1;
  KalmanGainInv(trkErr,hitErr,inverse_temp);
  KalmanGain(trkErr,inverse_temp,kGain);


 #pragma omp parallel for num_threads(bsize)
  for (size_t it=0;it<bsize;++it) {
    const float xin = x1(inPar,it);
    const float yin = _y1(inPar,it);
    const float zin = z1(inPar,it);
    const float ptin = 1.0f/ipt1(inPar,it);
    const float phiin = phi1(inPar,it);
    const float thetain = theta1(inPar,it);
    const float xout = x_pos1(msP,it);
    const float yout = y_pos1(msP,it);
    const float zout = z_pos1(msP,it);
  
    float xnew = xin + (kGain->data[0*bsize+it]*(xout-xin)) +(kGain->data[1*bsize+it]*(yout-yin));
    float ynew = yin + (kGain->data[3*bsize+it]*(xout-xin)) +(kGain->data[4*bsize+it]*(yout-yin));
    float znew = zin + (kGain->data[6*bsize+it]*(xout-xin)) +(kGain->data[7*bsize+it]*(yout-yin));
    float ptnew = ptin + (kGain->data[9*bsize+it]*(xout-xin)) +(kGain->data[10*bsize+it]*(yout-yin));
    float phinew = phiin + (kGain->data[12*bsize+it]*(xout-xin)) +(kGain->data[13*bsize+it]*(yout-yin));
    float thetanew = thetain + (kGain->data[15*bsize+it]*(xout-xin)) +(kGain->data[16*bsize+it]*(yout-yin));

    newErr->data[0*bsize+it] = trkErr->data[0*bsize+it] - (kGain->data[0*bsize+it]*trkErr->data[0*bsize+it]+kGain->data[1*bsize+it]*trkErr->data[1*bsize+it]+kGain->data[2*bsize+it]*trkErr->data[2*bsize+it]);
    newErr->data[1*bsize+it] = trkErr->data[1*bsize+it] - (kGain->data[0*bsize+it]*trkErr->data[1*bsize+it]+kGain->data[1*bsize+it]*trkErr->data[6*bsize+it]+kGain->data[2*bsize+it]*trkErr->data[7*bsize+it]);
    newErr->data[2*bsize+it] = trkErr->data[2*bsize+it] - (kGain->data[0*bsize+it]*trkErr->data[2*bsize+it]+kGain->data[1*bsize+it]*trkErr->data[7*bsize+it]+kGain->data[2*bsize+it]*trkErr->data[11*bsize+it]);
    newErr->data[3*bsize+it] = trkErr->data[3*bsize+it] - (kGain->data[0*bsize+it]*trkErr->data[3*bsize+it]+kGain->data[1*bsize+it]*trkErr->data[8*bsize+it]+kGain->data[2*bsize+it]*trkErr->data[12*bsize+it]);
    newErr->data[4*bsize+it] = trkErr->data[4*bsize+it] - (kGain->data[0*bsize+it]*trkErr->data[4*bsize+it]+kGain->data[1*bsize+it]*trkErr->data[9*bsize+it]+kGain->data[2*bsize+it]*trkErr->data[13*bsize+it]);
    newErr->data[5*bsize+it] = trkErr->data[5*bsize+it] - (kGain->data[0*bsize+it]*trkErr->data[5*bsize+it]+kGain->data[1*bsize+it]*trkErr->data[10*bsize+it]+kGain->data[2*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr->data[6*bsize+it] = trkErr->data[6*bsize+it] - (kGain->data[3*bsize+it]*trkErr->data[1*bsize+it]+kGain->data[4*bsize+it]*trkErr->data[6*bsize+it]+kGain->data[5*bsize+it]*trkErr->data[7*bsize+it]);
    newErr->data[7*bsize+it] = trkErr->data[7*bsize+it] - (kGain->data[3*bsize+it]*trkErr->data[2*bsize+it]+kGain->data[4*bsize+it]*trkErr->data[7*bsize+it]+kGain->data[5*bsize+it]*trkErr->data[11*bsize+it]);
    newErr->data[8*bsize+it] = trkErr->data[8*bsize+it] - (kGain->data[3*bsize+it]*trkErr->data[3*bsize+it]+kGain->data[4*bsize+it]*trkErr->data[8*bsize+it]+kGain->data[5*bsize+it]*trkErr->data[12*bsize+it]);
    newErr->data[9*bsize+it] = trkErr->data[9*bsize+it] - (kGain->data[3*bsize+it]*trkErr->data[4*bsize+it]+kGain->data[4*bsize+it]*trkErr->data[9*bsize+it]+kGain->data[5*bsize+it]*trkErr->data[13*bsize+it]);
    newErr->data[10*bsize+it] = trkErr->data[10*bsize+it] - (kGain->data[3*bsize+it]*trkErr->data[5*bsize+it]+kGain->data[4*bsize+it]*trkErr->data[10*bsize+it]+kGain->data[5*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr->data[11*bsize+it] = trkErr->data[11*bsize+it] - (kGain->data[6*bsize+it]*trkErr->data[2*bsize+it]+kGain->data[7*bsize+it]*trkErr->data[7*bsize+it]+kGain->data[8*bsize+it]*trkErr->data[11*bsize+it]);
    newErr->data[12*bsize+it] = trkErr->data[12*bsize+it] - (kGain->data[6*bsize+it]*trkErr->data[3*bsize+it]+kGain->data[7*bsize+it]*trkErr->data[8*bsize+it]+kGain->data[8*bsize+it]*trkErr->data[12*bsize+it]);
    newErr->data[13*bsize+it] = trkErr->data[13*bsize+it] - (kGain->data[6*bsize+it]*trkErr->data[4*bsize+it]+kGain->data[7*bsize+it]*trkErr->data[9*bsize+it]+kGain->data[8*bsize+it]*trkErr->data[13*bsize+it]);
    newErr->data[14*bsize+it] = trkErr->data[14*bsize+it] - (kGain->data[6*bsize+it]*trkErr->data[5*bsize+it]+kGain->data[7*bsize+it]*trkErr->data[10*bsize+it]+kGain->data[8*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr->data[15*bsize+it] = trkErr->data[15*bsize+it] - (kGain->data[9*bsize+it]*trkErr->data[3*bsize+it]+kGain->data[10*bsize+it]*trkErr->data[8*bsize+it]+kGain->data[11*bsize+it]*trkErr->data[12*bsize+it]);
    newErr->data[16*bsize+it] = trkErr->data[16*bsize+it] - (kGain->data[9*bsize+it]*trkErr->data[4*bsize+it]+kGain->data[10*bsize+it]*trkErr->data[9*bsize+it]+kGain->data[11*bsize+it]*trkErr->data[13*bsize+it]);
    newErr->data[17*bsize+it] = trkErr->data[17*bsize+it] - (kGain->data[9*bsize+it]*trkErr->data[5*bsize+it]+kGain->data[10*bsize+it]*trkErr->data[10*bsize+it]+kGain->data[11*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr->data[18*bsize+it] = trkErr->data[18*bsize+it] - (kGain->data[12*bsize+it]*trkErr->data[4*bsize+it]+kGain->data[13*bsize+it]*trkErr->data[9*bsize+it]+kGain->data[14*bsize+it]*trkErr->data[13*bsize+it]);
    newErr->data[19*bsize+it] = trkErr->data[19*bsize+it] - (kGain->data[12*bsize+it]*trkErr->data[5*bsize+it]+kGain->data[13*bsize+it]*trkErr->data[10*bsize+it]+kGain->data[14*bsize+it]*trkErr->data[14*bsize+it]);
  
    newErr->data[20*bsize+it] = trkErr->data[20*bsize+it] - (kGain->data[15*bsize+it]*trkErr->data[5*bsize+it]+kGain->data[16*bsize+it]*trkErr->data[10*bsize+it]+kGain->data[17*bsize+it]*trkErr->data[14*bsize+it]);

    setx1(inPar,it,xnew );
    sety1(inPar,it,ynew );
    setz1(inPar,it,znew);
    setipt1(inPar,it, ptnew);
    setphi1(inPar,it, phinew);
    settheta1(inPar,it, thetanew);
  }
 #pragma omp parallel for num_threads(bsize)
  for (size_t it=0;it<bsize;++it) {
    #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr->data[ i*bsize+it] = trkErr->data[ i*bsize+it] - newErr->data[ i*bsize+it];
    }
  }
}
#pragma omp end declare target

#pragma omp declare target
void KalmanUpdate_v2(struct MP6x6SF* trkErr, struct MP6F* inPar, const struct MP3x3SF* hitErr, const struct MP3F* msP, struct MP2x2SF* resErr_loc, struct MP2x6* kGain, struct MP2F* res_loc, struct MP6x6SF* newErr){

   // AddIntoUpperLeft2x2(psErr, msErr, resErr);
#pragma omp parallel for num_threads(bsize)
   for (size_t it=0;it<bsize;++it)
   {
     resErr_loc->data[0*bsize+it] = trkErr->data[0*bsize+it] + hitErr->data[0*bsize+it];
     resErr_loc->data[1*bsize+it] = trkErr->data[1*bsize+it] + hitErr->data[1*bsize+it];
     resErr_loc->data[2*bsize+it] = trkErr->data[2*bsize+it] + hitErr->data[2*bsize+it];
   }

   // Matriplex::InvertCramerSym(resErr);
#pragma omp parallel for num_threads(bsize)
   for (size_t it=0;it<bsize;++it)
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
#pragma omp parallel for num_threads(bsize)
   for (size_t it=0;it<bsize;++it)
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
#pragma omp parallel for num_threads(bsize)
   for (size_t it=0;it<bsize;++it)
   {
     res_loc->data[0*bsize+it] =  x_pos1(msP,it) - x1(inPar,it);
     res_loc->data[1*bsize+it] =  y_pos1(msP,it) - _y1(inPar,it);

     setx1    (inPar, it, x1    (inPar, it) + kGain->data[ 0*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 1*bsize+it] * res_loc->data[ 1*bsize+it]);
     sety1    (inPar, it, _y1    (inPar, it) + kGain->data[ 2*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 3*bsize+it] * res_loc->data[ 1*bsize+it]);
     setz1    (inPar, it, z1    (inPar, it) + kGain->data[ 4*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 5*bsize+it] * res_loc->data[ 1*bsize+it]);
     setipt1  (inPar, it, ipt1  (inPar, it) + kGain->data[ 6*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 7*bsize+it] * res_loc->data[ 1*bsize+it]);
     setphi1  (inPar, it, phi1  (inPar, it) + kGain->data[ 8*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[ 9*bsize+it] * res_loc->data[ 1*bsize+it]);
     settheta1(inPar, it, theta1(inPar, it) + kGain->data[10*bsize+it] * res_loc->data[ 0*bsize+it] + kGain->data[11*bsize+it] * res_loc->data[ 1*bsize+it]);
     //note: if ipt changes sign we should update the charge, or we should get rid of the charge altogether and just use the sign of ipt
   }

   // squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
   // missing

   // KHC(K, psErr, outErr);
   // outErr.Subtract(psErr, outErr);
#pragma omp parallel for num_threads(bsize)
   for (size_t it=0;it<bsize;++it)
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

#pragma omp parallel for num_threads(bsize)
  for (size_t it=0;it<bsize;++it) {
    #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr->data[ i*bsize+it] = trkErr->data[ i*bsize+it] - newErr->data[ i*bsize+it];
    }
  }
}
#pragma omp end declare target

//const float kfact = 100/3.8;
#define kfact 100./(-0.299792458*3.8112)
#pragma omp declare target
void propagateToZ(const struct MP6x6SF* inErr, const struct MP6F* inPar,
		  const struct MP1I* inChg, const struct MP3F* msP,
	                struct MP6x6SF* outErr, struct MP6F* outPar,
 		struct MP6x6F* errorProp, struct MP6x6F* temp) {
  //
 #pragma omp parallel for num_threads(bsize)
  for (size_t it=0;it<bsize;++it) {	
    const float zout = z_pos1(msP,it);
    const float k = q(inChg,it)*kfact;//100/3.8;
    const float deltaZ = zout - z1(inPar,it);
    const float pt = 1.0f/ipt1(inPar,it);
    const float cosP = cosf(phi1(inPar,it));
    const float sinP = sinf(phi1(inPar,it));
    const float cosT = cosf(theta1(inPar,it));
    const float sinT = sinf(theta1(inPar,it));
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0f/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*ipt1(inPar,it)*icosTk;
    //const float alpha = deltaZ*sinT*ipt1(inPar,it)/(cosT*k);
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;
    setx1(outPar,it, x1(inPar,it) + k*(pxin*sina - pyin*(1.0f-cosa)) );
    sety1(outPar,it, _y1(inPar,it) + k*(pyin*sina + pxin*(1.0f-cosa)) );
    setz1(outPar,it,zout);
    setipt1(outPar,it, ipt1(inPar,it));
    setphi1(outPar,it, phi1(inPar,it)+alpha );
    settheta1(outPar,it, theta1(inPar,it) );
    
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
    errorProp->data[bsize*PosInMtrx(4,2,6) + it] = -ipt1(inPar,it)*sinT*(icosTk);
    errorProp->data[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ*(icosTk);
    errorProp->data[bsize*PosInMtrx(4,5,6) + it] = ipt1(inPar,it)*deltaZ*(icosT*icosTk);
  }
  //
  MultHelixPropEndcap(errorProp, inErr, temp);
  MultHelixPropTranspEndcap(errorProp, temp, outErr);
}
#pragma omp end declare target
void memcpy_host2dev(struct MPTRK* trk, struct MPHIT* hit, int chunkSize, int lastChunkSize) {
    int localChunkSize = chunkSize;
    for (int s = 0; s<num_streams;s++){
		if( s==num_streams-1 ) {
			localChunkSize = lastChunkSize;
		}
     	#pragma omp target update to(trk[s*chunkSize*nb:localChunkSize*nb], hit[s*chunkSize*nb*nlayer:localChunkSize*nb*nlayer]) nowait depend(out:trk[s*chunkSize*nb:1])
	}
}

void memcpy_dev2host(struct MPTRK* outtrk, int chunkSize, int lastChunkSize) {
    int localChunkSize = chunkSize;
    for (int s = 0; s<num_streams;s++){
		if( s==num_streams-1 ) {
			localChunkSize = lastChunkSize;
		}
   		#pragma omp target update from(outtrk[s*chunkSize*nb:localChunkSize*nb]) nowait depend(inout:outtrk[s*chunkSize*nb:1])
	}
}

int main (int argc, char* argv[]) {

#if include_data == 1
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
   long start2, end2;
   double setup_time;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
#ifdef FIXED_RSEED
   //[DEBUG by Seyong on Dec. 28, 2020] add an explicit srand(1) call to generate fixed inputs for better debugging.
   srand(1);
#endif
   struct MPTRK* trk = prepareTracks(inputtrk);
   struct MPHIT* hit = prepareHits(inputhits);
   struct MPTRK* outtrk = (struct MPTRK*) malloc(nevts*nb*sizeof(struct MPTRK));

#pragma omp target enter data map(alloc: trk[0:nevts*nb], hit[0:nevts*nb*nlayer], outtrk[0:nevts*nb])

   int chunkSize = nevts/num_streams;
   int lastChunkSize = chunkSize;
   if( nevts%num_streams != 0 ) {
     lastChunkSize = chunkSize + (nevts - num_streams*chunkSize);
   }
#if include_data == 0
   memcpy_host2dev(trk, hit, chunkSize, lastChunkSize);
   #pragma omp taskwait
#endif

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   setup_time = ((double)(setup_stop - setup_start))*0.001;

   printf("done preparing!\n");
   
   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nlayer*nevts*nb*sizeof(struct MPHIT));
   printf("Size of struct MPTRK = %ld\n", sizeof(struct MPTRK));

   gettimeofday(&timecheck, NULL);
   start2 = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   int itr;
   for(itr=0; itr<NITER; itr++) {
     int localChunkSize = chunkSize;
     for(int s=0; s<num_streams; s++) {
		if( s==num_streams-1 ) {
			localChunkSize = lastChunkSize;
		}
#if include_data == 1
     	#pragma omp target update to(trk[s*chunkSize*nb:localChunkSize*nb], hit[s*chunkSize*nb*nlayer:localChunkSize*nb*nlayer]) nowait depend(out:trk[s*chunkSize*nb:1])
#endif
		//#pragma openarc cuda sharedRW(ie, ib)
#ifdef _OPENARC_
     	#pragma omp target teams distribute collapse(2) map(present,to: trk[s*chunkSize*nb:localChunkSize*nb], hit[s*chunkSize*nb*nlayer:localChunkSize*nb*nlayer]) map(present,from: outtrk[s*chunkSize*nb:localChunkSize*nb]) nowait depend(in:trk[s*chunkSize*nb:1]) depend(out:outtrk[s*chunkSize*nb:1])
#else
     	#pragma omp target teams distribute collapse(2) map(to: trk[s*chunkSize*nb:localChunkSize*nb], hit[s*chunkSize*nb*nlayer:localChunkSize*nb*nlayer]) map(from: outtrk[s*chunkSize*nb:localChunkSize*nb]) nowait depend(in:trk[s*chunkSize*nb:1]) depend(out:outtrk[s*chunkSize*nb:1])
#endif
     	for (size_t ie=s*chunkSize;ie<(s*chunkSize+localChunkSize);++ie) { // loop over events
       		for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
		 		//[DEBUG on Dec. 8, 2020] Moved gang-private variable declarations out of the device functions (propagateToZ and KalmanUpdate) to here.
   	     		struct MP6x6F errorProp, temp;
				struct MP2x2SF resErr_loc;
				struct MP2x6 kGain;
				struct MP2F res_loc;
  		 		struct MP6x6SF newErr;
         		const struct MPTRK* btracks = bTkC(trk, ie, ib);
         		struct MPTRK* obtracks = bTk(outtrk, ie, ib);
				(*obtracks) = (*btracks);
         		for(size_t layer=0; layer<nlayer; ++layer) {
            		const struct MPHIT* bhits = bHit4(hit, ie, ib,layer);
         		//
            		propagateToZ(&(*obtracks).cov, &(*obtracks).par, &(*obtracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par,
  	        		&errorProp, &temp); // vectorized function
            		//KalmanUpdate(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos, &inverse_temp, &kGain, &newErr);
                    KalmanUpdate_v2(&(*obtracks).cov, &(*obtracks).par, &(*bhits).cov, &(*bhits).pos, &resErr_loc, &kGain, &res_loc, &newErr);
         		}
       		}
     	}
#if include_data == 1
   		#pragma omp target update from(outtrk[s*chunkSize*nb:localChunkSize*nb]) nowait depend(inout:outtrk[s*chunkSize*nb:1])
#endif
	}  
   } //end of itr loop

   #pragma omp taskwait
   gettimeofday(&timecheck, NULL);
   end2 = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
#if include_data == 0
   memcpy_dev2host(outtrk, chunkSize, lastChunkSize);
   #pragma omp taskwait
#endif

   double wall_time = ((double)(end2 - start2))*0.001;
   printf("setup time time=%f (s)\n", setup_time);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*((int)NITER), wall_time, wall_time/(nevts*ntrks*((int)NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",((int)NITER),nevts, ntrks, bsize, nb, wall_time, setup_time, -1);
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
       float x_ = x3(outtrk,ie,it);
       float y_ = y3(outtrk,ie,it);
       float z_ = z3(outtrk,ie,it);
       float pt_ = 1./ipt3(outtrk,ie,it);
       float phi_ = phi3(outtrk,ie,it);
       float theta_ = theta3(outtrk,ie,it);
       float hx_ = x_pos3(hit,ie,it);
       float hy_ = y_pos3(hit,ie,it);
       float hz_ = z_pos3(hit,ie,it);
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
   avgpt = avgpt/((double)nevts*ntrks);
   avgphi = avgphi/((double)nevts*ntrks);
   avgtheta = avgtheta/((double)nevts*ntrks);
   avgx = avgx/((double)nevts*ntrks);
   avgy = avgy/((double)nevts*ntrks);
   avgz = avgz/((double)nevts*ntrks);
   avgdx = avgdx/((double)nevts*ntrks);
   avgdy = avgdy/((double)nevts*ntrks);
   avgdz = avgdz/((double)nevts*ntrks);

   double stdx = 0, stdy = 0, stdz = 0;
   double stddx = 0, stddy = 0, stddz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x3(outtrk,ie,it);
       float y_ = y3(outtrk,ie,it);
       float z_ = z3(outtrk,ie,it);
       float hx_ = x_pos3(hit,ie,it);
       float hy_ = y_pos3(hit,ie,it);
       float hz_ = z_pos3(hit,ie,it);
       float pt_ = 1./ipt3(outtrk,ie,it);
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
       x_ = x3(trk,ie,it);
       y_ = y3(trk,ie,it);
       z_ = z3(trk,ie,it);
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

   stdx = sqrtf(stdx/((double)nevts*ntrks));
   stdy = sqrtf(stdy/((double)nevts*ntrks));
   stdz = sqrtf(stdz/((double)nevts*ntrks));
   stddx = sqrtf(stddx/((double)nevts*ntrks));
   stddy = sqrtf(stddy/((double)nevts*ntrks));
   stddz = sqrtf(stddz/((double)nevts*ntrks));

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

 #pragma omp target exit data map(delete: trk[0:nevts*nb], hit[0:nevts*nb*nlayer], outtrk[0:nevts*nb])
   free(trk);
   free(hit);
   free(outtrk);

   return 0;
}
