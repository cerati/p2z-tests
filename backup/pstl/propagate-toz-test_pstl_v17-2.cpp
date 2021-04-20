/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <execution>

#ifndef bsize
#define bsize 128
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    ntrks/bsize
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

#ifdef __NVCOMPILER_CUDA__

#include <thrust/iterator/counting_iterator.h>
using namespace thrust;

#else //X86

#include <tbb/tbb.h>
using namespace tbb;

#endif


inline size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

const std::array<size_t, 36> SymOffsets66{0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};

struct ATRK {
  std::array<float,6> par;
  std::array<float,21> cov;
  int q;
};

struct AHIT {
  std::array<float,3> pos;
  std::array<float,6> cov;
};

//#define VEC

#ifndef __NVCOMPILER_CUDA__

template <typename T, int N, int base>
struct MPNX {
   std::array<T,N*base> data;
};

#else

enum class IPAR {X = 0, Y = 1, Z = 2, Ipt = 3, Phi = 4, Theta = 5};

constexpr int iparX     = 0;
constexpr int iparY     = 1;
constexpr int iparZ     = 2;
constexpr int iparIpt   = 3;
constexpr int iparPhi   = 4;
constexpr int iparTheta = 5;

template <typename T, int N, int base>
struct MPNX {
   using DataType = T;

   static constexpr int N_    = N;
   static constexpr int base_ = base;

   std::vector<T> data;

   MPNX()                           : data(N*base){}
   MPNX(const size_t els)           : data(N*base*els){}
   MPNX(const std::vector<T> data_) : data(data_){}

   int GetN()    const {return N;}
   int GetBase() const {return base;}  
};

#endif

using MP1I    = MPNX<int,   1 , bsize>;//MPTRK.q
//using MP22I   = MPNX<int,   22, bsize>;//?
using MP3F    = MPNX<float, 3 , bsize>;//MPHIT.pos
using MP6F    = MPNX<float, 6 , bsize>;//MPTRK.par
using MP3x3   = MPNX<float, 9 , bsize>;//inverse_temp=>
using MP3x6   = MPNX<float, 18, bsize>;//kGain
using MP3x3SF = MPNX<float, 6 , bsize>;//MPHIT.cov
using MP6x6SF = MPNX<float, 21, bsize>;//MPTRK.cov, newErr
using MP6x6F  = MPNX<float, 36, bsize>;//errorProp, temp


#ifdef __NVCOMPILER_CUDA__

template <typename MPNTp>
struct MPNXAccessor {
   typedef typename MPNTp::DataType T;

   static constexpr size_t basesz = MPNTp::base_;
   static constexpr size_t N      = MPNTp::N_;
   static constexpr size_t stride = N*basesz;

   T* data_; //accessor field only for the LL data access, not allocated here

   MPNXAccessor() : data_(nullptr) {}
   MPNXAccessor(const MPNTp &v) : data_(const_cast<T*>(v.data.data())){
	}

   T* operator()(const size_t i = 0) const {return (data_ + stride*i);}
   T  operator()(const size_t i, const size_t j) const {return (data_ + stride*i)[j];}
   T  operator[](const size_t i) const {return data_[i];}

   template <int ipar, typename AccessedFieldTp = MPNTp>
   typename std::enable_if<(std::is_same<AccessedFieldTp, MP3F>::value and ipar < 3) or (std::is_same<AccessedFieldTp, MP6F>::value and ipar < 6), T>::type
   Get(size_t it, size_t id)  const { return (data_ + stride*id)[it + ipar*basesz]; }

   template <int ipar, typename AccessedFieldTp = MPNTp> 
   typename std::enable_if<std::is_same<AccessedFieldTp, MP3F>::value and ipar < 3, void>::type
   Set(size_t it, float val, size_t id)    { (data_ + stride*id)[it + ipar*basesz] = val; }

   template <int ipar, typename AccessedFieldTp = MPNTp>
   typename std::enable_if<(std::is_same<AccessedFieldTp, MP3F>::value and ipar < 3) or (std::is_same<AccessedFieldTp, MP6F>::value and ipar < 6), T>::type
   Get(size_t it)  const { return data_[it + ipar*basesz]; }

   template <int ipar, typename AccessedFieldTp = MPNTp> 
   typename std::enable_if<std::is_same<AccessedFieldTp, MP3F>::value and ipar < 3, void>::type
   Set(size_t it, T val)    { data_[it + ipar*basesz] = val; }

   // same as above but with a (shifted) raw pointer (and more generic)
   template <int ipar>
   static T Get(const T* local_data, size_t it)  { return local_data[it + ipar*basesz]; }  

   template <int ipar>
   static void Set(T* local_data, size_t it, T val)     { local_data[it + ipar*basesz] = val; }  

};

using MP6FAccessor   = MPNXAccessor<MP6F>;
using MP6x6SFAccessor= MPNXAccessor<MP6x6SF>;
using MP1IAccessor   = MPNXAccessor<MP1I>;

using MP3FAccessor   = MPNXAccessor<MP3F>;
using MP3x3SFAccessor= MPNXAccessor<MP3x3SF>;

using MP6x6FAccessor= MPNXAccessor<MP6x6F>;
using MP3x3Accessor = MPNXAccessor<MP3x3>;
using MP3x6Accessor = MPNXAccessor<MP3x6>;

#endif

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;
#ifdef __NVCOMPILER_CUDA__
  MPTRK() : par(), cov(), q() {}
  MPTRK(const size_t els) : par(els), cov(els), q(els) {}
#endif
  //  MP22I   hitidx;
};

#ifdef __NVCOMPILER_CUDA__
struct MPTRKAccessor {
  MP6FAccessor    par;
  MP6x6SFAccessor cov;
  MP1IAccessor    q;
  MPTRKAccessor() : par(), cov(), q() {}
  MPTRKAccessor(const MPTRK &in) : par(in.par), cov(in.cov), q(in.q) {}
};
#endif

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
#ifdef __NVCOMPILER_CUDA__
  MPHIT() : pos(), cov(){}
  MPHIT(const size_t els) : pos(els), cov(els) {}
#endif
};

#ifdef __NVCOMPILER_CUDA__
struct MPHITAccessor {
  MP3FAccessor    pos;
  MP3x3SFAccessor cov;
  MPHITAccessor() : pos(), cov() {}
  MPHITAccessor(const MPHIT &in) : pos(in.pos), cov(in.cov) {}
};
#endif


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

inline const MPTRK* bTkC(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

inline float q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
inline float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
inline float x    (const MP6F* bpars, size_t it){ return par(bpars, it, 0); }
inline float y    (const MP6F* bpars, size_t it){ return par(bpars, it, 1); }
inline float z    (const MP6F* bpars, size_t it){ return par(bpars, it, 2); }
inline float ipt  (const MP6F* bpars, size_t it){ return par(bpars, it, 3); }
inline float phi  (const MP6F* bpars, size_t it){ return par(bpars, it, 4); }
inline float theta(const MP6F* bpars, size_t it){ return par(bpars, it, 5); }
//
inline float par(const MPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
inline float x    (const MPTRK* btracks, size_t it){ return par(btracks, it, 0); }
inline float y    (const MPTRK* btracks, size_t it){ return par(btracks, it, 1); }
inline float z    (const MPTRK* btracks, size_t it){ return par(btracks, it, 2); }
inline float ipt  (const MPTRK* btracks, size_t it){ return par(btracks, it, 3); }
inline float phi  (const MPTRK* btracks, size_t it){ return par(btracks, it, 4); }
inline float theta(const MPTRK* btracks, size_t it){ return par(btracks, it, 5); }
//
inline float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTkC(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
inline float x    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
inline float y    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
inline float z    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
inline float ipt  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
inline float phi  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
inline float theta(const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
inline void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
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

inline const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
inline const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib, size_t lay) {
  return &(hits[lay + (ib + nb*ev)*nlayer]);
}
//
inline float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
inline float x(const MP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
inline float y(const MP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
inline float z(const MP3F* hpos, size_t it)    { return pos(hpos, it, 2); }
//
inline float pos(const MPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
inline float x(const MPHIT* hits, size_t it)    { return pos(hits, it, 0); }
inline float y(const MPHIT* hits, size_t it)    { return pos(hits, it, 1); }
inline float z(const MPHIT* hits, size_t it)    { return pos(hits, it, 2); }
//
float pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
inline float x(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
inline float y(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
inline float z(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }

MPTRK* prepareTracks(struct ATRK inputtrk) {
#ifndef __NVCOMPILER_CUDA__
  MPTRK* result = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK)); //fixme, align?
#else
  MPTRK* result = new MPTRK[nevts*nb];
#endif
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
    	//par
    	for (size_t ip=0;ip<6;++ip) {
    	  result[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
    	}
    	//cov
    	for (size_t ip=0;ip<21;++ip) {
    	  result[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
    	}
    	//q
    	result[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      }
    }
  }
  return result;
}

#ifdef __NVCOMPILER_CUDA__
std::shared_ptr<MPTRK> prepareTracksN(struct ATRK inputtrk) {

  auto result = std::make_shared<MPTRK>(nevts*nb);

  // store in element order for bunches of bsize matrices (a la matriplex)
  const size_t stride_par = bsize*6;
  const size_t stride_cov = bsize*21;
  const size_t stride_q   = bsize*1;
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      const int l = ib + nb*ie; 
      for (size_t it=0;it<bsize;++it) {
    	//par
    	for (size_t ip=0;ip<6;++ip) {
    	  result->par.data[l*stride_par + ip*bsize + it] = (1+smear*randn(0,1))*inputtrk.par[ip];
    	}
    	//cov
    	for (size_t ip=0;ip<21;++ip) {
    	  result->cov.data[l*stride_cov + ip*bsize + it] = (1+smear*randn(0,1))*inputtrk.cov[ip];
    	}
    	//q
    	result->q.data[l*stride_q + it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      }
    }
  }
  return std::move(result);
}
#endif

MPHIT* prepareHits(struct AHIT inputhit) {
#ifndef __NVCOMPILER_CUDA__
  MPHIT* result = (MPHIT*) malloc(nlayer*nevts*nb*sizeof(MPHIT));  //fixme, align?
#else
  MPHIT* result = new MPHIT[nlayer*nevts*nb];
#endif
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
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

#ifdef __NVCOMPILER_CUDA__
std::shared_ptr<MPHIT> prepareHitsN(struct AHIT inputhit) {
  auto result = std::make_shared<MPHIT>(nlayer*nevts*nb);
  // store in element order for bunches of bsize matrices (a la matriplex)
  const size_t stride_pos = bsize*3;
  const size_t stride_cov = bsize*6;

  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        const size_t l = ib + nb*ie;
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  result->pos.data[(lay+nlayer*l)*stride_pos + ip*bsize + it] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  result->cov.data[(lay+nlayer*l)*stride_cov + ip*bsize +it] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return std::move(result);
}
#endif

#define N bsize 
template <size_t block_size = 1>
void MultHelixPropEndcap(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C, const size_t offset = 0) {
  const auto &a = A->data; //ASSUME_ALIGNED(a, 64);
  const auto &b = B->data; //ASSUME_ALIGNED(b, 64);
  auto &c = C->data;       //ASSUME_ALIGNED(c, 64);
#pragma simd
  for (int n = offset; n < N; n += block_size)
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

template <size_t block_size = 1>
void MultHelixPropTranspEndcap(const MP6x6F* A, const MP6x6F* B, MP6x6SF* C, const size_t offset = 0) {
  const auto &a = A->data; //ASSUME_ALIGNED(a, 64);
  const auto &b = B->data; //ASSUME_ALIGNED(b, 64);
  auto &c = C->data;       //ASSUME_ALIGNED(c, 64);
#pragma simd
  for (int n = offset; n < N; n += block_size)
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

template <size_t block_size = 1>
void KalmanGainInv(const MP6x6SF* A, const MP3x3SF* B, MP3x3* C, const size_t offset = 0) {
  const auto &a = A->data; //ASSUME_ALIGNED(a, 64);
  const auto &b = B->data; //ASSUME_ALIGNED(b, 64);
  auto &c = C->data;       //ASSUME_ALIGNED(c, 64);
#pragma simd
  for (int n = offset; n < N; n += block_size)
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

template <size_t block_size = 1>
void KalmanGain(const MP6x6SF* A, const MP3x3* B, MP3x6* C, const size_t offset = 0) {
  const auto &a = A->data; //ASSUME_ALIGNED(a, 64);
  const auto &b = B->data; //ASSUME_ALIGNED(b, 64);
  auto &c = C->data;       //ASSUME_ALIGNED(c, 64);
#pragma simd  
  for (int n = offset; n < N; n += block_size)
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

template <size_t block_size = 1>
void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP, MP3x3* inverse_temp, MP3x6* kGain, MP6x6SF* newErr, const size_t offset = 0){

  KalmanGainInv<block_size>(trkErr,hitErr,inverse_temp, offset);
  KalmanGain<block_size>(trkErr,inverse_temp,kGain, offset);

#pragma simd
  for (size_t it=offset;it<bsize; it += block_size) {
    const float xin = x(inPar,it);
    const float yin = y(inPar,it);
    const float zin = z(inPar,it);
    const float ptin = 1./ipt(inPar,it);
    const float phiin = phi(inPar,it);
    const float thetain = theta(inPar,it);
    const float xout = x(msP,it);
    const float yout = y(msP,it);
    const float zout = z(msP,it);
  
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

    setx(inPar,it,xnew );
    sety(inPar,it,ynew );
    setz(inPar,it,znew);
    setipt(inPar,it, ptnew);
    setphi(inPar,it, phinew);
    settheta(inPar,it, thetanew);
  }
  trkErr = newErr;
}

const float kfact = 100/3.8;
template <size_t block_size = 1>
void propagateToZ(const MP6x6SF* inErr, const MP6F* inPar,
		  const MP1I* inChg, const MP3F* msP,
	                MP6x6SF* outErr, MP6F* outPar,
			MP6x6F* errorProp, MP6x6F* temp, const size_t offset = 0) {
  //
  for (size_t it=offset;it<bsize; it += block_size) {	
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
    setx(outPar,it, x(inPar,it) + k*(pxin*sina - pyin*(1.-cosa)) );
    sety(outPar,it, y(inPar,it) + k*(pyin*sina + pxin*(1.-cosa)) );
    setz(outPar,it,zout);
    setipt(outPar,it, ipt(inPar,it));
    setphi(outPar,it, phi(inPar,it)+alpha );
    settheta(outPar,it, theta(inPar,it) );
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
    
    for (size_t i=0;i<6;++i) errorProp->data[bsize*PosInMtrx(i,i,6) + it] = 1.;
    errorProp->data[bsize*PosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp->data[bsize*PosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.-cCosPsina))*(pt*pt);
    errorProp->data[bsize*PosInMtrx(0,4,6) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp->data[bsize*PosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*icosT);
    errorProp->data[bsize*PosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp->data[bsize*PosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.-cCosPsina))*(pt*pt);
    errorProp->data[bsize*PosInMtrx(1,4,6) + it] = (k*pt)*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp->data[bsize*PosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp->data[bsize*PosInMtrx(4,2,6) + it] = -ipt(inPar,it)*sinT*(icosTk);
    errorProp->data[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ*(icosTk);
    errorProp->data[bsize*PosInMtrx(4,5,6) + it] = ipt(inPar,it)*deltaZ*(icosT*icosTk);
//    errorProp->data[bsize*PosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
//    errorProp->data[bsize*PosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*ipt(inPar,it))-k*(cosP*sina-sinP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
//    errorProp->data[bsize*PosInMtrx(0,4,6) + it] = (k/ipt(inPar,it))*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
//    errorProp->data[bsize*PosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
//    errorProp->data[bsize*PosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
//    errorProp->data[bsize*PosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*ipt(inPar,it))-k*(sinP*sina+cosP*(1.-cCosPsina))/(ipt(inPar,it)*ipt(inPar,it));
//    errorProp->data[bsize*PosInMtrx(1,4,6) + it] = (k/ipt(inPar,it))*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
//    errorProp->data[bsize*PosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
//    errorProp->data[bsize*PosInMtrx(4,2,6) + it] = -ipt(inPar,it)*sinT/(cosT*k);
//    errorProp->data[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ/(cosT*k);
//    errorProp->data[bsize*PosInMtrx(4,5,6) + it] = ipt(inPar,it)*deltaZ/(cosT*cosT*k);
  }
  //
  MultHelixPropEndcap<block_size>(errorProp, inErr, temp, offset);
  MultHelixPropTranspEndcap<block_size>(errorProp, temp, outErr, offset);
}

/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////

template <size_t block_size = 1>
void MultHelixPropEndcap_mod(const MP6x6FAccessor &A, const MP6x6SFAccessor &B, MP6x6FAccessor &C, const size_t lid, const size_t offset = 0) {
  const auto a = A(lid); 
  const auto b = B(lid); 
  auto c = C(lid);      
#pragma simd
  for (int n = offset; n < N; n += block_size)
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

template <size_t block_size = 1>
void MultHelixPropTranspEndcap_mod(const MP6x6FAccessor &A, const MP6x6FAccessor &B, MP6x6SFAccessor &C, const size_t lid, const size_t offset = 0) {
  const auto a = A(lid);
  const auto b = B(lid);
  auto c = C(lid);     
#pragma simd
  for (int n = offset; n < N; n += block_size)
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


template <size_t block_size = 1>
void MultHelixPropTranspEndcap_mod2(const MP6x6FAccessor &A, MP6x6SFAccessor &B, const size_t lid, const size_t offset = 0) {
  const auto a = A(lid);
  auto b = B(lid);
#pragma simd
  for (int n = offset; n < N; n += block_size)
  {
    float temp00 = b[ 0*N+n] + a[ 2*N+n]*b[ 3*N+n] + a[ 3*N+n]*b[ 6*N+n] + a[ 4*N+n]*b[10*N+n] + a[ 5*N+n]*b[15*N+n];
    //float temp01 = b[ 1*N+n] + a[ 2*N+n]*b[ 4*N+n] + a[ 3*N+n]*b[ 7*N+n] + a[ 4*N+n]*b[11*N+n] + a[ 5*N+n]*b[16*N+n];
    float temp02 = b[ 3*N+n] + a[ 2*N+n]*b[ 5*N+n] + a[ 3*N+n]*b[ 8*N+n] + a[ 4*N+n]*b[12*N+n] + a[ 5*N+n]*b[17*N+n];
    float temp03 = b[ 6*N+n] + a[ 2*N+n]*b[ 8*N+n] + a[ 3*N+n]*b[ 9*N+n] + a[ 4*N+n]*b[13*N+n] + a[ 5*N+n]*b[18*N+n];
    float temp04 = b[10*N+n] + a[ 2*N+n]*b[12*N+n] + a[ 3*N+n]*b[13*N+n] + a[ 4*N+n]*b[14*N+n] + a[ 5*N+n]*b[19*N+n];
    float temp05 = b[15*N+n] + a[ 2*N+n]*b[17*N+n] + a[ 3*N+n]*b[18*N+n] + a[ 4*N+n]*b[19*N+n] + a[ 5*N+n]*b[20*N+n];
    float temp06 = b[ 1*N+n] + a[ 8*N+n]*b[ 3*N+n] + a[ 9*N+n]*b[ 6*N+n] + a[10*N+n]*b[10*N+n] + a[11*N+n]*b[15*N+n];
    float temp07 = b[ 2*N+n] + a[ 8*N+n]*b[ 4*N+n] + a[ 9*N+n]*b[ 7*N+n] + a[10*N+n]*b[11*N+n] + a[11*N+n]*b[16*N+n];
    float temp08 = b[ 4*N+n] + a[ 8*N+n]*b[ 5*N+n] + a[ 9*N+n]*b[ 8*N+n] + a[10*N+n]*b[12*N+n] + a[11*N+n]*b[17*N+n];
    float temp09 = b[ 7*N+n] + a[ 8*N+n]*b[ 8*N+n] + a[ 9*N+n]*b[ 9*N+n] + a[10*N+n]*b[13*N+n] + a[11*N+n]*b[18*N+n];
    float temp10 = b[11*N+n] + a[ 8*N+n]*b[12*N+n] + a[ 9*N+n]*b[13*N+n] + a[10*N+n]*b[14*N+n] + a[11*N+n]*b[19*N+n];
    float temp11 = b[16*N+n] + a[ 8*N+n]*b[17*N+n] + a[ 9*N+n]*b[18*N+n] + a[10*N+n]*b[19*N+n] + a[11*N+n]*b[20*N+n];
    //float temp12 = 0;
    //float temp13 = 0;
    //float temp14 = 0;
    //float temp15 = 0;
    //float temp16 = 0;
    //float temp17 = 0;
    float temp18 = b[ 6*N+n];
    float temp19 = b[ 7*N+n];
    float temp20 = b[ 8*N+n];
    float temp21 = b[ 9*N+n];
    float temp22 = b[13*N+n];
    float temp23 = b[18*N+n];
    float temp24 = a[26*N+n]*b[ 3*N+n] + a[27*N+n]*b[ 6*N+n] + b[10*N+n] + a[29*N+n]*b[15*N+n];
    float temp25 = a[26*N+n]*b[ 4*N+n] + a[27*N+n]*b[ 7*N+n] + b[11*N+n] + a[29*N+n]*b[16*N+n];
    float temp26 = a[26*N+n]*b[ 5*N+n] + a[27*N+n]*b[ 8*N+n] + b[12*N+n] + a[29*N+n]*b[17*N+n];
    float temp27 = a[26*N+n]*b[ 8*N+n] + a[27*N+n]*b[ 9*N+n] + b[13*N+n] + a[29*N+n]*b[18*N+n];
    float temp28 = a[26*N+n]*b[12*N+n] + a[27*N+n]*b[13*N+n] + b[14*N+n] + a[29*N+n]*b[19*N+n];
    float temp29 = a[26*N+n]*b[17*N+n] + a[27*N+n]*b[18*N+n] + b[19*N+n] + a[29*N+n]*b[20*N+n];
    float temp30 = b[15*N+n];
    float temp31 = b[16*N+n];
    float temp32 = b[17*N+n];
    float temp33 = b[18*N+n];
    float temp34 = b[19*N+n];
    float temp35 = b[20*N+n];

    b[ 0*N+n] = temp00 + temp02*a[ 2*N+n] + temp03*a[ 3*N+n] + temp04*a[ 4*N+n] + temp05*a[ 5*N+n];
    b[ 1*N+n] = temp06 + temp08*a[ 2*N+n] + temp09*a[ 3*N+n] + temp10*a[ 4*N+n] + temp11*a[ 5*N+n];
    b[ 2*N+n] = temp07 + temp08*a[ 8*N+n] + temp09*a[ 9*N+n] + temp10*a[10*N+n] + temp11*a[11*N+n];
    b[ 3*N+n] = 0;
    b[ 4*N+n] = 0;
    b[ 5*N+n] = 0;
    b[ 6*N+n] = temp18 + temp20*a[ 2*N+n] + temp21*a[ 3*N+n] + temp22*a[ 4*N+n] + temp23*a[ 5*N+n];
    b[ 7*N+n] = temp19 + temp20*a[ 8*N+n] + temp21*a[ 9*N+n] + temp22*a[10*N+n] + temp23*a[11*N+n];
    b[ 8*N+n] = 0;
    b[ 9*N+n] = temp21;
    b[10*N+n] = temp24 + temp26*a[ 2*N+n] + temp27*a[ 3*N+n] + temp28*a[ 4*N+n] + temp29*a[ 5*N+n];
    b[11*N+n] = b[25*N+n] + temp26*a[ 8*N+n] + temp27*a[ 9*N+n] + temp28*a[10*N+n] + temp29*a[11*N+n];
    b[12*N+n] = 0;
    b[13*N+n] = temp27;
    b[14*N+n] = temp26*a[26*N+n] + temp27*a[27*N+n] + temp28 + temp29*a[29*N+n];
    b[15*N+n] = temp30 + temp32*a[ 2*N+n] + temp33*a[ 3*N+n] + temp34*a[ 4*N+n] + temp35*a[ 5*N+n];
    b[16*N+n] = temp31 + temp32*a[ 8*N+n] + temp33*a[ 9*N+n] + temp34*a[10*N+n] + temp35*a[11*N+n];
    b[17*N+n] = 0;
    b[18*N+n] = temp33;
    b[19*N+n] = temp32*a[26*N+n] + temp33*a[27*N+n] + temp34 + temp35*a[29*N+n];
    b[20*N+n] = temp35;
  }
}

template <size_t block_size = 1>
void KalmanUpdate_mod(MP6x6SFAccessor  &trkErrAcc,
                      MP6FAccessor &inParAcc, 
		      const MPHITAccessor &bhits, 
                      const size_t lid, 
                      const size_t llid, 
                      const size_t offset = 0) {

  const auto trkErr   = trkErrAcc(lid);
  auto inPar          = inParAcc (lid);

  const auto hitErr   = bhits.cov(llid);
  const auto msP      = bhits.pos(llid);
 
#pragma simd
  for (size_t it=offset; it<bsize; it += block_size) {
    const float xin     = MP6FAccessor::Get<iparX>(inPar, it);
    const float yin     = MP6FAccessor::Get<iparY>(inPar, it);
    const float zin     = MP6FAccessor::Get<iparZ>(inPar, it);
    const float ptin    = 1./MP6FAccessor::Get<iparIpt>(inPar, it);
    const float phiin   = MP6FAccessor::Get<iparPhi>(inPar, it);
    const float thetain = MP6FAccessor::Get<iparTheta>(inPar, it);
    const float xout = MP3FAccessor::Get<iparX>(msP, it);
    const float yout = MP3FAccessor::Get<iparY>(msP, it);
    const float zout = MP3FAccessor::Get<iparZ>(msP, it);


    double det =
      ((trkErr[0*bsize+it]+hitErr[0*bsize+it])*(((trkErr[ 6*bsize+it]+hitErr[ 3*bsize+it]) *(trkErr[11*bsize+it]+hitErr[5*bsize+it])) - ((trkErr[7*bsize+it]+hitErr[4*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])))) -
      ((trkErr[1*bsize+it]+hitErr[1*bsize+it])*(((trkErr[ 1*bsize+it]+hitErr[ 1*bsize+it]) *(trkErr[11*bsize+it]+hitErr[5*bsize+it])) - ((trkErr[7*bsize+it]+hitErr[4*bsize+it]) *(trkErr[2*bsize+it]+hitErr[2*bsize+it])))) +
      ((trkErr[2*bsize+it]+hitErr[2*bsize+it])*(((trkErr[ 1*bsize+it]+hitErr[ 1*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[6*bsize+it]+hitErr[3*bsize+it]))));
    double invdet = 1.0/det;

    float temp00 =  invdet*(((trkErr[ 6*bsize+it]+hitErr[ 3*bsize+it]) *(trkErr[11*bsize+it]+hitErr[5*bsize+it])) - ((trkErr[7*bsize+it]+hitErr[4*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])));
    float temp01 =  -1*invdet*(((trkErr[ 1*bsize+it]+hitErr[ 1*bsize+it]) *(trkErr[11*bsize+it]+hitErr[5*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])));
    float temp02 =  invdet*(((trkErr[ 1*bsize+it]+hitErr[ 1*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])));
    float temp03 =  -1*invdet*(((trkErr[ 1*bsize+it]+hitErr[ 1*bsize+it]) *(trkErr[11*bsize+it]+hitErr[5*bsize+it])) - ((trkErr[7*bsize+it]+hitErr[4*bsize+it]) *(trkErr[2*bsize+it]+hitErr[2*bsize+it])));
    float temp04 =  invdet*(((trkErr[ 0*bsize+it]+hitErr[ 0*bsize+it]) *(trkErr[11*bsize+it]+hitErr[5*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[2*bsize+it]+hitErr[2*bsize+it])));
    float temp05 =  -1*invdet*(((trkErr[ 0*bsize+it]+hitErr[ 0*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[1*bsize+it]+hitErr[1*bsize+it])));
    float temp06 =  invdet*(((trkErr[ 1*bsize+it]+hitErr[ 1*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[6*bsize+it]+hitErr[3*bsize+it])));
    float temp07 =  -1*invdet*(((trkErr[ 0*bsize+it]+hitErr[ 0*bsize+it]) *(trkErr[7*bsize+it]+hitErr[4*bsize+it])) - ((trkErr[2*bsize+it]+hitErr[2*bsize+it]) *(trkErr[1*bsize+it]+hitErr[1*bsize+it])));
    float temp08 =  invdet*(((trkErr[ 0*bsize+it]+hitErr[ 0*bsize+it]) *(trkErr[6*bsize+it]+hitErr[3*bsize+it])) - ((trkErr[1*bsize+it]+hitErr[1*bsize+it]) *(trkErr[1*bsize+it]+hitErr[1*bsize+it])));
    //
    float temp09 = 0.0f;

    float kGain00 = trkErr[0*bsize+it]*temp00 + trkErr[1*bsize+it]*temp03 + trkErr[2*bsize+it]*temp06;
    float kGain01 = trkErr[0*bsize+it]*temp01 + trkErr[1*bsize+it]*temp04 + trkErr[2*bsize+it]*temp07;
    float kGain02 = trkErr[0*bsize+it]*temp02 + trkErr[1*bsize+it]*temp05 + trkErr[2*bsize+it]*temp08;
    float kGain03 = trkErr[1*bsize+it]*temp00 + trkErr[6*bsize+it]*temp03 + trkErr[7*bsize+it]*temp06;
    float kGain04 = trkErr[1*bsize+it]*temp01 + trkErr[6*bsize+it]*temp04 + trkErr[7*bsize+it]*temp07;
    float kGain05 = trkErr[1*bsize+it]*temp02 + trkErr[6*bsize+it]*temp05 + trkErr[7*bsize+it]*temp08;
    float kGain06 = trkErr[2*bsize+it]*temp00 + trkErr[7*bsize+it]*temp03 + trkErr[11*bsize+it]*temp06;
    float kGain07 = trkErr[2*bsize+it]*temp01 + trkErr[7*bsize+it]*temp04 + trkErr[11*bsize+it]*temp07;
    float kGain08 = trkErr[2*bsize+it]*temp02 + trkErr[7*bsize+it]*temp05 + trkErr[11*bsize+it]*temp08;
    float kGain09 = trkErr[3*bsize+it]*temp00 + trkErr[8*bsize+it]*temp03 + trkErr[12*bsize+it]*temp06;
    float kGain10 = trkErr[3*bsize+it]*temp01 + trkErr[8*bsize+it]*temp04 + trkErr[12*bsize+it]*temp07;
    float kGain11 = trkErr[3*bsize+it]*temp02 + trkErr[8*bsize+it]*temp05 + trkErr[12*bsize+it]*temp08;
    float kGain12 = trkErr[4*bsize+it]*temp00 + trkErr[9*bsize+it]*temp03 + trkErr[13*bsize+it]*temp06;
    float kGain13 = trkErr[4*bsize+it]*temp01 + trkErr[9*bsize+it]*temp04 + trkErr[13*bsize+it]*temp07;
    float kGain14 = trkErr[4*bsize+it]*temp02 + trkErr[9*bsize+it]*temp05 + trkErr[13*bsize+it]*temp08;
    float kGain15 = trkErr[5*bsize+it]*temp00 + trkErr[10*bsize+it]*temp03 + trkErr[14*bsize+it]*temp06;
    float kGain16 = trkErr[5*bsize+it]*temp01 + trkErr[10*bsize+it]*temp04 + trkErr[14*bsize+it]*temp07;
    float kGain17 = trkErr[5*bsize+it]*temp02 + trkErr[10*bsize+it]*temp05 + trkErr[14*bsize+it]*temp08;
  
    float xnew = xin + (kGain00*(xout-xin)) +(kGain01*(yout-yin));
    float ynew = yin + (kGain03*(xout-xin)) +(kGain04*(yout-yin));
    float znew = zin + (kGain06*(xout-xin)) +(kGain07*(yout-yin));
    float ptnew = ptin + (kGain09*(xout-xin)) +(kGain10*(yout-yin));
    float phinew = phiin + (kGain12*(xout-xin)) +(kGain13*(yout-yin));
    float thetanew = thetain + (kGain15*(xout-xin)) +(kGain16*(yout-yin));

    temp00 = trkErr[0*bsize+it] - (kGain00*trkErr[0*bsize+it]+kGain01*trkErr[1*bsize+it]+kGain02*trkErr[2*bsize+it]);

    trkErr[0*bsize+it] = temp00;

    temp00 = trkErr[1*bsize+it] - (kGain00*trkErr[1*bsize+it]+kGain01*trkErr[6*bsize+it]+kGain02*trkErr[7*bsize+it]);
    temp01 = trkErr[2*bsize+it] - (kGain00*trkErr[2*bsize+it]+kGain01*trkErr[7*bsize+it]+kGain02*trkErr[11*bsize+it]);
    temp02 = trkErr[3*bsize+it] - (kGain00*trkErr[3*bsize+it]+kGain01*trkErr[8*bsize+it]+kGain02*trkErr[12*bsize+it]);
    temp03 = trkErr[4*bsize+it] - (kGain00*trkErr[4*bsize+it]+kGain01*trkErr[9*bsize+it]+kGain02*trkErr[13*bsize+it]);
    temp04 = trkErr[5*bsize+it] - (kGain00*trkErr[5*bsize+it]+kGain01*trkErr[10*bsize+it]+kGain02*trkErr[14*bsize+it]);
  
    temp05 = trkErr[6*bsize+it] - (kGain03*trkErr[1*bsize+it]+kGain04*trkErr[6*bsize+it]+kGain05*trkErr[7*bsize+it]);

    trkErr[1*bsize+it] = temp00;
    trkErr[6*bsize+it] = temp05;

    temp00 = trkErr[7*bsize+it] - (kGain03*trkErr[2*bsize+it]+kGain04*trkErr[7*bsize+it]+kGain05*trkErr[11*bsize+it]);
    temp05 = trkErr[8*bsize+it] - (kGain03*trkErr[3*bsize+it]+kGain04*trkErr[8*bsize+it]+kGain05*trkErr[12*bsize+it]);
    temp06 = trkErr[9*bsize+it] - (kGain03*trkErr[4*bsize+it]+kGain04*trkErr[9*bsize+it]+kGain05*trkErr[13*bsize+it]);
    temp07 = trkErr[10*bsize+it] - (kGain03*trkErr[5*bsize+it]+kGain04*trkErr[10*bsize+it]+kGain05*trkErr[14*bsize+it]);
  
    temp08 = trkErr[11*bsize+it] - (kGain06*trkErr[2*bsize+it]+kGain07*trkErr[7*bsize+it]+kGain08*trkErr[11*bsize+it]);

    trkErr[2*bsize+it]  = temp01;
    trkErr[7*bsize+it]  = temp00;
    trkErr[11*bsize+it] = temp08;

    temp01 = trkErr[12*bsize+it] - (kGain06*trkErr[3*bsize+it]+kGain07*trkErr[8*bsize+it]+kGain08*trkErr[12*bsize+it]);
    temp00 = trkErr[13*bsize+it] - (kGain06*trkErr[4*bsize+it]+kGain07*trkErr[9*bsize+it]+kGain08*trkErr[13*bsize+it]);
    temp08 = trkErr[14*bsize+it] - (kGain06*trkErr[5*bsize+it]+kGain07*trkErr[10*bsize+it]+kGain08*trkErr[14*bsize+it]);
    temp09 = trkErr[15*bsize+it] - (kGain09*trkErr[3*bsize+it]+kGain10*trkErr[8*bsize+it]+kGain11*trkErr[12*bsize+it]);

    trkErr[3*bsize+it]  = temp02;
    trkErr[8*bsize+it]  = temp05;
    trkErr[12*bsize+it] = temp01;
    trkErr[15*bsize+it] = temp09;

    temp02 = trkErr[16*bsize+it] - (kGain09*trkErr[4*bsize+it]+kGain10*trkErr[9*bsize+it]+kGain11*trkErr[13*bsize+it]);

    trkErr[16*bsize+it] = temp02;

    temp05 = trkErr[17*bsize+it] - (kGain09*trkErr[5*bsize+it]+kGain10*trkErr[10*bsize+it]+kGain11*trkErr[14*bsize+it]);

    trkErr[17*bsize+it] = temp05;
  
    temp01 = trkErr[18*bsize+it] - (kGain12*trkErr[4*bsize+it]+kGain13*trkErr[9*bsize+it]+kGain14*trkErr[13*bsize+it]);

    trkErr[4*bsize+it]  = temp03;
    trkErr[9*bsize+it]  = temp06;
    trkErr[13*bsize+it] = temp00;
    trkErr[18*bsize+it] = temp01;

    temp09 = trkErr[19*bsize+it] - (kGain12*trkErr[5*bsize+it]+kGain13*trkErr[10*bsize+it]+kGain14*trkErr[14*bsize+it]);

    trkErr[19*bsize+it] = temp09;

    temp09 = trkErr[20*bsize+it] - (kGain15*trkErr[5*bsize+it]+kGain16*trkErr[10*bsize+it]+kGain17*trkErr[14*bsize+it]);

    trkErr[10*bsize+it] = temp07;
    trkErr[5*bsize+it]  = temp04;
    trkErr[14*bsize+it] = temp08;
    trkErr[20*bsize+it] = temp09;

    MP6FAccessor::Set<iparX>(inPar,it, xnew);
    MP6FAccessor::Set<iparY>(inPar,it, ynew);
    MP6FAccessor::Set<iparZ>(inPar,it, znew);
    MP6FAccessor::Set<iparIpt>(inPar,it, ptnew);
    MP6FAccessor::Set<iparPhi>(inPar,it, phinew);
    MP6FAccessor::Set<iparTheta>(inPar,it, thetanew);
  }

  return;
}



template <size_t block_size = 1>
void propagateToZ_mod(const MPTRKAccessor &btracks, 
		      const MPHITAccessor &bhits,
                      MPTRKAccessor  &obtracks, 
		      MP6x6FAccessor &errorPropAcc, 
                      MP6x6FAccessor &tempAcc, 
                      const size_t lid, 
                      const size_t llid, 
                      const size_t offset = 0) {

  const auto inPar    = btracks.par(lid);
  const auto inChg    = btracks.q  (lid);

  const auto msP      = bhits.pos(llid);

  auto outErr    = obtracks.cov(lid); 
  auto outPar    = obtracks.par(lid); 
  auto errorProp = errorPropAcc(lid);
#pragma simd
  for (size_t it=offset;it<bsize; it += block_size) {	
    const float zout = MP3FAccessor::Get<iparZ>(msP, it);
    const float k    = inChg[it]*kfact;//100/3.8;
    const float deltaZ = zout - MP6FAccessor::Get<iparZ>(inPar, it);
    const float pt   = 1. / MP6FAccessor::Get<iparIpt>(inPar, it);
    const float cosP = cosf(MP6FAccessor::Get<iparPhi>(inPar, it));
    const float sinP = sinf(MP6FAccessor::Get<iparPhi>(inPar, it));
    const float cosT = cosf(MP6FAccessor::Get<iparTheta>(inPar, it));
    const float sinT = sinf(MP6FAccessor::Get<iparTheta>(inPar, it));

    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*MP6FAccessor::Get<iparIpt>(inPar, it)*icosTk;

    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;
    MP6FAccessor::Set<iparX>(outPar,it, MP6FAccessor::Get<iparX>(inPar, it) + k*(pxin*sina - pyin*(1.-cosa)) );
    MP6FAccessor::Set<iparY>(outPar,it, MP6FAccessor::Get<iparY>(inPar, it) + k*(pyin*sina + pxin*(1.-cosa)) );
    MP6FAccessor::Set<iparZ>(outPar,it, zout);
    MP6FAccessor::Set<iparIpt>(outPar,it, MP6FAccessor::Get<iparIpt>(inPar, it));
    MP6FAccessor::Set<iparPhi>(outPar,it, MP6FAccessor::Get<iparPhi>(inPar, it)+alpha );
    MP6FAccessor::Set<iparTheta>(outPar,it, MP6FAccessor::Get<iparTheta>(inPar, it) );
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
#pragma unroll    
    for (size_t i=0;i<6;++i) errorProp[bsize*PosInMtrx(i,i,6) + it] = 1.;

    errorProp[bsize*PosInMtrx(0,2,6) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp[bsize*PosInMtrx(0,3,6) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.-cCosPsina))*(pt*pt);
    errorProp[bsize*PosInMtrx(0,4,6) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp[bsize*PosInMtrx(0,5,6) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*icosT);
    errorProp[bsize*PosInMtrx(1,2,6) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp[bsize*PosInMtrx(1,3,6) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.-cCosPsina))*(pt*pt);
    errorProp[bsize*PosInMtrx(1,4,6) + it] = (k*pt)*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp[bsize*PosInMtrx(1,5,6) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp[bsize*PosInMtrx(4,2,6) + it] = -MP6FAccessor::Get<iparIpt>(inPar, it)*sinT*(icosTk);
    errorProp[bsize*PosInMtrx(4,3,6) + it] = sinT*deltaZ*(icosTk);
    errorProp[bsize*PosInMtrx(4,5,6) + it] = MP6FAccessor::Get<iparIpt>(inPar, it)*deltaZ*(icosT*icosTk);
  }

  //MultHelixPropEndcap_mod<block_size>(errorPropAcc, btracks.cov, tempAcc, lid, offset);
  //MultHelixPropTranspEndcap_mod<block_size>(errorPropAcc, tempAcc, obtracks.cov, lid, offset);//obtracks.cov => trkErr

  MultHelixPropTranspEndcap_mod2<block_size>(errorPropAcc, obtracks.cov, lid, offset);

  return;
}


int main (int argc, char* argv[]) {

   int itr;
   struct ATRK inputtrk = {
     {-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975},
     {6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
      6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,
      0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348},
     1
   };

   struct AHIT inputhit = {
     {-20.7824649810791, -12.24150276184082, 57.8067626953125},
     {2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945}
   };

   printf("track in pos: %f, %f, %f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2]);
   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66[PosInMtrx(0,0,6)]],
	                                       inputtrk.cov[SymOffsets66[PosInMtrx(1,1,6)]],
	                                       inputtrk.cov[SymOffsets66[PosInMtrx(2,2,6)]]);
   printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);
   
   long setup_start, setup_stop;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   
   MPTRK* trk = prepareTracks(inputtrk); 
   MPHIT* hit = prepareHits(inputhit);

   MPTRK* outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));

   auto trkNPtr = prepareTracksN(inputtrk);
   std::unique_ptr<MPTRKAccessor> trkNaccPtr(new MPTRKAccessor(*trkNPtr));

   auto hitNPtr = prepareHitsN(inputhit);
   std::unique_ptr<MPHITAccessor> hitNaccPtr(new MPHITAccessor(*hitNPtr));

   std::unique_ptr<MPTRK> outtrkNPtr(new MPTRK(nevts*nb));
   std::unique_ptr<MPTRKAccessor> outtrkNaccPtr(new MPTRKAccessor(*outtrkNPtr));

   std::unique_ptr<MP6x6F>  errPropPtr(new MP6x6F(nevts*nb));
   std::unique_ptr<MP6x6FAccessor>  errorPropAccPtr(new MP6x6FAccessor(*errPropPtr));

   std::unique_ptr<MP6x6F>  tempPtr(new MP6x6F(nevts*nb));
   std::unique_ptr<MP6x6FAccessor>  tempAccPtr(new MP6x6FAccessor(*tempPtr));

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");
   
   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   auto wall_start = std::chrono::high_resolution_clock::now();

   constexpr size_t blk_sz = 1;
   auto policy = std::execution::par_unseq;

   for(itr=0; itr<NITER; itr++) {

     const int outer_loop_range = nevts*nb*blk_sz;
     const int nbxblk_sz        = nb*blk_sz;

     std::for_each(policy,
                   counting_iterator(0),
                   counting_iterator(outer_loop_range),
                   [=,  &trkNacc = *trkNaccPtr, 
			&hitNacc = *hitNaccPtr, 
			&outtrkNacc = *outtrkNaccPtr,
			&errorPropAcc = *errorPropAccPtr,
			&tempAcc = *tempAccPtr] (auto ii) {
                   const size_t ie = ii / nbxblk_sz;
                   const size_t ibt= ii - ie*nbxblk_sz;
                   const size_t ib = ibt / blk_sz;  
                   const size_t inner_loop_offset = ibt - ib*blk_sz;
                  
                   for(size_t layer=0; layer<nlayer; ++layer) {
                     const size_t lii = layer+ii*nlayer;
                     //
                     //propagateToZ<blk_sz>(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par,
                     //&errorProp, &temp, inner_loop_offset); // vectorized function
                     //KalmanUpdate<blk_sz>(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos, &inverse_temp, &kGain, &newErr, inner_loop_offset);
                     propagateToZ_mod<blk_sz>(trkNacc, hitNacc, outtrkNacc, errorPropAcc, tempAcc, ii, lii, inner_loop_offset);
                     KalmanUpdate_mod<blk_sz>(outtrkNacc.cov, outtrkNacc.par, hitNacc, ii, lii, inner_loop_offset);
                   }

                   });
   } //end of itr loop

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);

exit(-1);

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
#ifndef __NVCOMPILER_CUDA__
   free(trk);
   free(hit);
   free(outtrk);
#else
   delete [] trk;
   delete [] hit;
   delete [] outtrk;
#endif

   return 0;
}
