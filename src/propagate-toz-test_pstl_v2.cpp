/*
nvc++ -O2 -std=c++17 -stdpar=gpu -gpu=cc75 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll  src/propagate-tor-test_pstl.cpp   -o ./propagate_nvcpp_pstl
nvc++ -O2 -std=c++17 -stdpar=multicore src/propagate-tor-test_pstl.cpp   -o ./propagate_nvcpp_pstl 
g++ -O3 -I. -fopenmp -mavx512f -std=c++17 src/propagate-tor-test_pstl.cpp -lm -lgomp -Lpath-to-tbb-lib -ltbb  -o ./propagate_gcc_pstl
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <ranges>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <execution>
#include <random>

#if defined(__NVCOMPILER_CUDA__)
#define USE_PARALLEL_COPY
#endif

#if defined(__NVCOMPILER_CUDA__)
constexpr bool use_cuda = true;
#else
constexpr bool use_cuda = false;
#endif//__NVCOMPILER_CUDA__

#ifndef ntrks
#define ntrks 9600//8192
#endif

#ifndef bsize
#define bsize 32
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

#ifdef include_data
constexpr bool include_data_transfer = true;
#else
constexpr bool include_data_transfer = false;
#endif

using namespace std::ranges; 

auto PosInMtrx = [](const size_t &&i, const size_t &&j, const size_t &&D, const size_t block_size = 1) constexpr {return block_size*(i*D+j);};

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

constexpr int iparX     = 0;
constexpr int iparY     = 1;
constexpr int iparZ     = 2;
constexpr int iparIpt   = 3;
constexpr int iparPhi   = 4;
constexpr int iparTheta = 5;

template <typename T, int N, int bSize = 1>
struct MPNX {
   std::array<T,N*bSize> data;

   MPNX() = default;
   MPNX(const MPNX<T, N, bSize> &) = default;
   MPNX(MPNX<T, N, bSize> &&)      = default;
   
   //basic accessors
   constexpr T &operator[](const int i) { return data[i]; }
   constexpr const T &operator[](const int i) const { return data[i]; }
   constexpr T& operator()(const int i, const int j) {return data[i*bSize+j];}
   constexpr const T& operator()(const int i, const int j) const {return data[i*bSize+j];}

   constexpr int size() const { return N*bSize; }   
   //
   inline void load(MPNX<T, N, 1>& dst, const int b) const {
#pragma unroll
     for (int ip=0;ip<N;++ip) {   	
    	dst.data[ip] = data[ip*bSize + b]; 
     }
     
     return;
   }

   inline void save(const MPNX<T, N, 1>& src, const int b) {
#pragma unroll
     for (int ip=0;ip<N;++ip) {    	
    	 data[ip*bSize + b] = src.data[ip]; 
     }
     
     return;
   }  

   auto operator=(const MPNX&) -> MPNX& = default;
   auto operator=(MPNX&&     ) -> MPNX& = default;
};

// internal data formats (coinside with external ones for x86):
template<int bSize = 1> using MP1I_    = MPNX<int,   1 , bSize>;
template<int bSize = 1> using MP1F_    = MPNX<float, 1 , bSize>;
template<int bSize = 1> using MP2F_    = MPNX<float, 2 , bSize>;
template<int bSize = 1> using MP3F_    = MPNX<float, 3 , bSize>;
template<int bSize = 1> using MP6F_    = MPNX<float, 6 , bSize>;
template<int bSize = 1> using MP2x2SF_ = MPNX<float, 3 , bSize>;
template<int bSize = 1> using MP3x3SF_ = MPNX<float, 6 , bSize>;
template<int bSize = 1> using MP6x6SF_ = MPNX<float, 21, bSize>;
template<int bSize = 1> using MP6x6F_  = MPNX<float, 36, bSize>;
template<int bSize = 1> using MP3x3_   = MPNX<float, 9 , bSize>;
template<int bSize = 1> using MP3x6_   = MPNX<float, 18, bSize>;
template<int bSize = 1> using MP2x6_   = MPNX<float, 12, bSize>;

// external data formats:
using MP1I    = MPNX<int,   1 , bsize>;
using MP1F    = MPNX<float, 1 , bsize>;
using MP2F    = MPNX<float, 2 , bsize>;
using MP3F    = MPNX<float, 3 , bsize>;
using MP6F    = MPNX<float, 6 , bsize>;
using MP2x2SF = MPNX<float, 3 , bsize>;
using MP3x3SF = MPNX<float, 6 , bsize>;
using MP6x6SF = MPNX<float, 21, bsize>;
using MP6x6F  = MPNX<float, 36, bsize>;
using MP3x3   = MPNX<float, 9 , bsize>;
using MP3x6   = MPNX<float, 18, bsize>;

template <int N = 1>
struct MPTRK_ {
  MP6F_<N>    par;
  MP6x6SF_<N> cov;
  MP1I_<N>    q;
};

template <int N = 1>
struct MPHIT_ {
  MP3F_<N>    pos;
  MP3x3SF_<N> cov;
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK() = default;
  //
  template<int S>
  inline decltype(auto) load(const int batch_id = 0) const{
  
    MPTRK_<S> dst;

    if constexpr (std::is_same<MP6F, MP6F_<S>>::value        
                  and std::is_same<MP6x6SF, MP6x6SF_<S>>::value
                  and std::is_same<MP1I, MP1I_<S>>::value)  { //just do a copy of the whole objects
      dst.par = this->par;
      dst.cov = this->cov;
      dst.q   = this->q;
      
    } else { //ok, do manual load of the batch component instead
      this->par.load(dst.par, batch_id);
      this->cov.load(dst.cov, batch_id);
      this->q.load(dst.q, batch_id);
    }//done
    
    return dst;  
  }
  
  template<int S>
  inline void save(MPTRK_<S> &src, const int batch_id = 0) {
  
    if constexpr (std::is_same<MP6F, MP6F_<S>>::value        
                  and std::is_same<MP6x6SF, MP6x6SF_<S>>::value
                  and std::is_same<MP1I, MP1I_<S>>::value) { //just do a copy of the whole objects
      this->par = src.par;
      this->cov = src.cov;
      this->q   = src.q;

    } else { //ok, do manual load of the batch component instead
      this->par.save(src.par, batch_id);
      this->cov.save(src.cov, batch_id);
      this->q.save(src.q, batch_id);
    }//done
    
    return;
  } 
////
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;

  MPHIT() = default;

  template<int S>
  inline decltype(auto) load(const int batch_id = 0) const {
    MPHIT_<S> dst;
    
    if constexpr (std::is_same<MP3F, MP3F_<S>>::value        
                  and std::is_same<MP3x3SF, MP3x3SF_<S>>::value) { //just do a copy of the whole object
      dst.pos = this->pos;
      dst.cov = this->cov;
    } else { //ok, do manual load of the batch component instead
      this->pos.load(dst.pos, batch_id);
      this->cov.load(dst.cov, batch_id);
    }//done    
    
    return dst;
  }
////
};

///////////////////////////////////////
//Gen. utils

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

void prepareTracks(std::vector<MPTRK> &trcks, ATRK &inputtrk) {
  //
  auto fill_trck = [=, &inputtrk=inputtrk](auto&& trck) {
  
                       for (auto&& it : views::iota(0,bsize)) {
	                 //par
	                 for (auto&& ip : views::iota(0,6) ) {
	                   trck.par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
	                 }
	                 //cov, scale by factor 100
	                 for (auto&& ip : views::iota(0,21)) {
	                   trck.cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
	                 }
	                 //q
	                 trck.q.data[it] = inputtrk.q;//can't really smear this or fit will be wrong
                       }
                       
                   };
                      
  for_each(trcks, fill_trck); 
  //
  return;
}

void prepareHits(std::vector<MPHIT> &hits, AHIT* inputhits) {
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (auto&& lay : iota_view{0,nlayer}) {
    for_each(views::iota(0, nevts*nb), [=, &inputhit = inputhits[lay], &hits = hits] (auto&& evtrk) {
      for (auto&& it : views::iota(0,bsize)) {
        //pos
        for (auto&& ip : views::iota(0,3)) {
          hits[lay+nlayer*evtrk].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        }
        //cov
        for (auto&& ip : views::iota(0,6)) {
          hits[lay+nlayer*evtrk].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        }
      }
    }); 
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////////////
// Aux utils 
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

const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib,size_t lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float Pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x(const MP3F* hpos, size_t it)    { return Pos(hpos, it, 0); }
float y(const MP3F* hpos, size_t it)    { return Pos(hpos, it, 1); }
float z(const MP3F* hpos, size_t it)    { return Pos(hpos, it, 2); }
//
float Pos(const MPHIT* hits, size_t it, size_t ipar){
  return Pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT* hits, size_t it)    { return Pos(hits, it, 0); }
float y(const MPHIT* hits, size_t it)    { return Pos(hits, it, 1); }
float z(const MPHIT* hits, size_t it)    { return Pos(hits, it, 2); }
//
float Pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT* bhits = bHit(hits, ev, ib, nlayer-1);
  size_t it = tk % bsize;
  return Pos(bhits,it,ipar);
}
float x(const MPHIT* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 0); }
float y(const MPHIT* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 1); }
float z(const MPHIT* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 2); }


////////////////////////////////////////////////////////////////////////
///MAIN compute kernels

template<int N = 1>
inline void MultHelixPropEndcap(const MP6x6F_<N> &a, const MP6x6SF_<N> &b, MP6x6F_<N> &c) {
#pragma unroll
 for (int n = 0; n < N; ++n) {
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
    c[12*N+n] = 0.f;
    c[13*N+n] = 0.f;
    c[14*N+n] = 0.f;
    c[15*N+n] = 0.f;
    c[16*N+n] = 0.f;
    c[17*N+n] = 0.f;
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
  return;
}

template<int N = 1>
inline void MultHelixPropTranspEndcap(const MP6x6F_<N> &a, const MP6x6F_<N> &b, MP6x6SF_<N> &c) {
#pragma unroll
  for (int n = 0; n < N; ++n) {
    c[ 0*N+n] = b[ 0*N+n] + b[ 2*N+n]*a[ 2*N+n] + b[ 3*N+n]*a[ 3*N+n] + b[ 4*N+n]*a[ 4*N+n] + b[ 5*N+n]*a[ 5*N+n];
    c[ 1*N+n] = b[ 6*N+n] + b[ 8*N+n]*a[ 2*N+n] + b[ 9*N+n]*a[ 3*N+n] + b[10*N+n]*a[ 4*N+n] + b[11*N+n]*a[ 5*N+n];
    c[ 2*N+n] = b[ 7*N+n] + b[ 8*N+n]*a[ 8*N+n] + b[ 9*N+n]*a[ 9*N+n] + b[10*N+n]*a[10*N+n] + b[11*N+n]*a[11*N+n];
    c[ 3*N+n] = b[12*N+n] + b[14*N+n]*a[ 2*N+n] + b[15*N+n]*a[ 3*N+n] + b[16*N+n]*a[ 4*N+n] + b[17*N+n]*a[ 5*N+n];
    c[ 4*N+n] = b[13*N+n] + b[14*N+n]*a[ 8*N+n] + b[15*N+n]*a[ 9*N+n] + b[16*N+n]*a[10*N+n] + b[17*N+n]*a[11*N+n];
    c[ 5*N+n] = 0.f;
    c[ 6*N+n] = b[18*N+n] + b[20*N+n]*a[ 2*N+n] + b[21*N+n]*a[ 3*N+n] + b[22*N+n]*a[ 4*N+n] + b[23*N+n]*a[ 5*N+n];
    c[ 7*N+n] = b[19*N+n] + b[20*N+n]*a[ 8*N+n] + b[21*N+n]*a[ 9*N+n] + b[22*N+n]*a[10*N+n] + b[23*N+n]*a[11*N+n];
    c[ 8*N+n] = 0.f;
    c[ 9*N+n] = b[21*N+n];
    c[10*N+n] = b[24*N+n] + b[26*N+n]*a[ 2*N+n] + b[27*N+n]*a[ 3*N+n] + b[28*N+n]*a[ 4*N+n] + b[29*N+n]*a[ 5*N+n];
    c[11*N+n] = b[25*N+n] + b[26*N+n]*a[ 8*N+n] + b[27*N+n]*a[ 9*N+n] + b[28*N+n]*a[10*N+n] + b[29*N+n]*a[11*N+n];
    c[12*N+n] = 0.f;
    c[13*N+n] = b[27*N+n];
    c[14*N+n] = b[26*N+n]*a[26*N+n] + b[27*N+n]*a[27*N+n] + b[28*N+n] + b[29*N+n]*a[29*N+n];
    c[15*N+n] = b[30*N+n] + b[32*N+n]*a[ 2*N+n] + b[33*N+n]*a[ 3*N+n] + b[34*N+n]*a[ 4*N+n] + b[35*N+n]*a[ 5*N+n];
    c[16*N+n] = b[31*N+n] + b[32*N+n]*a[ 8*N+n] + b[33*N+n]*a[ 9*N+n] + b[34*N+n]*a[10*N+n] + b[35*N+n]*a[11*N+n];
    c[17*N+n] = 0.f;
    c[18*N+n] = b[33*N+n];
    c[19*N+n] = b[32*N+n]*a[26*N+n] + b[33*N+n]*a[27*N+n] + b[34*N+n] + b[35*N+n]*a[29*N+n];
    c[20*N+n] = b[35*N+n];
  }
  return;
}

template<int N = 1>
inline void KalmanGainInv(const MP6x6SF_<N> &a, const MP3x3SF_<N> &b, MP3x3_<N> &c) {

#pragma unroll
  for (int n = 0; n < N; ++n) {
    double det =
      ((a[0*N+n]+b[0*N+n])*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])))) -
      ((a[1*N+n]+b[1*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])))) +
      ((a[2*N+n]+b[2*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n]))));
    double invdet = 1.0/det;

    c[ 0*N+n] =   invdet*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 1*N+n] =  -invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 2*N+n] =   invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 3*N+n] =  -invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 4*N+n] =   invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 5*N+n] =  -invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 6*N+n] =   invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n])));
    c[ 7*N+n] =  -invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 8*N+n] =   invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[6*N+n]+b[3*N+n])) - ((a[1*N+n]+b[1*N+n]) *(a[1*N+n]+b[1*N+n])));
  }
  
  return;
}

template <int N = 1>
inline void KalmanGain(const MP6x6SF_<N> &a, const MP3x3_<N> &b, MP3x6_<N> &c) {

#pragma unroll
  for (int n = 0; n < N; ++n) {
    c[ 0*N+n] = a[0*N+n]*b[0*N+n] + a[ 1*N+n]*b[3*N+n] + a[2*N+n]*b[6*N+n];
    c[ 1*N+n] = a[0*N+n]*b[1*N+n] + a[ 1*N+n]*b[4*N+n] + a[2*N+n]*b[7*N+n];
    c[ 2*N+n] = a[0*N+n]*b[2*N+n] + a[ 1*N+n]*b[5*N+n] + a[2*N+n]*b[8*N+n];
    c[ 3*N+n] = a[1*N+n]*b[0*N+n] + a[ 6*N+n]*b[3*N+n] + a[7*N+n]*b[6*N+n];
    c[ 4*N+n] = a[1*N+n]*b[1*N+n] + a[ 6*N+n]*b[4*N+n] + a[7*N+n]*b[7*N+n];
    c[ 5*N+n] = a[1*N+n]*b[2*N+n] + a[ 6*N+n]*b[5*N+n] + a[7*N+n]*b[8*N+n];
    c[ 6*N+n] = a[2*N+n]*b[0*N+n] + a[ 7*N+n]*b[3*N+n] + a[11*N+n]*b[6*N+n];
    c[ 7*N+n] = a[2*N+n]*b[1*N+n] + a[ 7*N+n]*b[4*N+n] + a[11*N+n]*b[7*N+n];
    c[ 8*N+n] = a[2*N+n]*b[2*N+n] + a[ 7*N+n]*b[5*N+n] + a[11*N+n]*b[8*N+n];
    c[ 9*N+n] = a[3*N+n]*b[0*N+n] + a[ 8*N+n]*b[3*N+n] + a[12*N+n]*b[6*N+n];
    c[10*N+n] = a[3*N+n]*b[1*N+n] + a[ 8*N+n]*b[4*N+n] + a[12*N+n]*b[7*N+n];
    c[11*N+n] = a[3*N+n]*b[2*N+n] + a[ 8*N+n]*b[5*N+n] + a[12*N+n]*b[8*N+n];
    c[12*N+n] = a[4*N+n]*b[0*N+n] + a[ 9*N+n]*b[3*N+n] + a[13*N+n]*b[6*N+n];
    c[13*N+n] = a[4*N+n]*b[1*N+n] + a[ 9*N+n]*b[4*N+n] + a[13*N+n]*b[7*N+n];
    c[14*N+n] = a[4*N+n]*b[2*N+n] + a[ 9*N+n]*b[5*N+n] + a[13*N+n]*b[8*N+n];
    c[15*N+n] = a[5*N+n]*b[0*N+n] + a[10*N+n]*b[3*N+n] + a[14*N+n]*b[6*N+n];
    c[16*N+n] = a[5*N+n]*b[1*N+n] + a[10*N+n]*b[4*N+n] + a[14*N+n]*b[7*N+n];
    c[17*N+n] = a[5*N+n]*b[2*N+n] + a[10*N+n]*b[5*N+n] + a[14*N+n]*b[8*N+n];
  }
  
  return;
}

template <int N = 1>
void KalmanUpdate(MP6x6SF_<N> &trkErr, MP6F_<N> &inPar, const MP3x3SF_<N> &hitErr, const MP3F_<N> &msP){

  MP3x3_<N> inverse_temp;
  MP3x6_<N> kGain;
  MP6x6SF_<N> newErr;
  
  KalmanGainInv<N>(trkErr, hitErr, inverse_temp);
  KalmanGain<N>(trkErr, inverse_temp, kGain);

//#pragma omp simd
  for (size_t it = 0;it < N;++it) {
    const auto xin     = inPar(iparX,it);
    const auto yin     = inPar(iparY,it);
    const auto zin     = inPar(iparZ,it);
    const auto ptin    = 1.f/ inPar(iparIpt,it);
    const auto phiin   = inPar(iparPhi,it);
    const auto thetain = inPar(iparTheta,it);
    const auto xout    = msP(iparX,it);
    const auto yout    = msP(iparY,it);
    //const auto zout    = msP(iparZ,it);

    auto xnew     = xin + (kGain[0*N+it]*(xout-xin)) +(kGain[1*N+it]*(yout-yin)); 
    auto ynew     = yin + (kGain[3*N+it]*(xout-xin)) +(kGain[4*N+it]*(yout-yin)); 
    auto znew     = zin + (kGain[6*N+it]*(xout-xin)) +(kGain[7*N+it]*(yout-yin)); 
    auto ptnew    = ptin + (kGain[9*N+it]*(xout-xin)) +(kGain[10*N+it]*(yout-yin)); 
    auto phinew   = phiin + (kGain[12*N+it]*(xout-xin)) +(kGain[13*N+it]*(yout-yin)); 
    auto thetanew = thetain + (kGain[15*N+it]*(xout-xin)) +(kGain[16*N+it]*(yout-yin)); 

    newErr[ 0*N+it] = trkErr[ 0*N+it] - (kGain[ 0*N+it]*trkErr[0*N+it]+kGain[1*N+it]*trkErr[1*N+it]+kGain[2*N+it]*trkErr[2*N+it]);
    newErr[ 1*N+it] = trkErr[ 1*N+it] - (kGain[ 0*N+it]*trkErr[1*N+it]+kGain[1*N+it]*trkErr[6*N+it]+kGain[2*N+it]*trkErr[7*N+it]);
    newErr[ 2*N+it] = trkErr[ 2*N+it] - (kGain[ 0*N+it]*trkErr[2*N+it]+kGain[1*N+it]*trkErr[7*N+it]+kGain[2*N+it]*trkErr[11*N+it]);
    newErr[ 3*N+it] = trkErr[ 3*N+it] - (kGain[ 0*N+it]*trkErr[3*N+it]+kGain[1*N+it]*trkErr[8*N+it]+kGain[2*N+it]*trkErr[12*N+it]);
    newErr[ 4*N+it] = trkErr[ 4*N+it] - (kGain[ 0*N+it]*trkErr[4*N+it]+kGain[1*N+it]*trkErr[9*N+it]+kGain[2*N+it]*trkErr[13*N+it]);
    newErr[ 5*N+it] = trkErr[ 5*N+it] - (kGain[ 0*N+it]*trkErr[5*N+it]+kGain[1*N+it]*trkErr[10*N+it]+kGain[2*N+it]*trkErr[14*N+it]);

    newErr[ 6*N+it] = trkErr[ 6*N+it] - (kGain[ 3*N+it]*trkErr[1*N+it]+kGain[4*N+it]*trkErr[6*N+it]+kGain[5*N+it]*trkErr[7*N+it]);
    newErr[ 7*N+it] = trkErr[ 7*N+it] - (kGain[ 3*N+it]*trkErr[2*N+it]+kGain[4*N+it]*trkErr[7*N+it]+kGain[5*N+it]*trkErr[11*N+it]);
    newErr[ 8*N+it] = trkErr[ 8*N+it] - (kGain[ 3*N+it]*trkErr[3*N+it]+kGain[4*N+it]*trkErr[8*N+it]+kGain[5*N+it]*trkErr[12*N+it]);
    newErr[ 9*N+it] = trkErr[ 9*N+it] - (kGain[ 3*N+it]*trkErr[4*N+it]+kGain[4*N+it]*trkErr[9*N+it]+kGain[5*N+it]*trkErr[13*N+it]);
    newErr[10*N+it] = trkErr[10*N+it] - (kGain[ 3*N+it]*trkErr[5*N+it]+kGain[4*N+it]*trkErr[10*N+it]+kGain[5*N+it]*trkErr[14*N+it]);

    newErr[11*N+it] = trkErr[11*N+it] - (kGain[ 6*N+it]*trkErr[2*N+it]+kGain[7*N+it]*trkErr[7*N+it]+kGain[8*N+it]*trkErr[11*N+it]);
    newErr[12*N+it] = trkErr[12*N+it] - (kGain[ 6*N+it]*trkErr[3*N+it]+kGain[7*N+it]*trkErr[8*N+it]+kGain[8*N+it]*trkErr[12*N+it]);
    newErr[13*N+it] = trkErr[13*N+it] - (kGain[ 6*N+it]*trkErr[4*N+it]+kGain[7*N+it]*trkErr[9*N+it]+kGain[8*N+it]*trkErr[13*N+it]);
    newErr[14*N+it] = trkErr[14*N+it] - (kGain[ 6*N+it]*trkErr[5*N+it]+kGain[7*N+it]*trkErr[10*N+it]+kGain[8*N+it]*trkErr[14*N+it]);

    newErr[15*N+it] = trkErr[15*N+it] - (kGain[ 9*N+it]*trkErr[3*N+it]+kGain[10*N+it]*trkErr[8*N+it]+kGain[11*N+it]*trkErr[12*N+it]);
    newErr[16*N+it] = trkErr[16*N+it] - (kGain[ 9*N+it]*trkErr[4*N+it]+kGain[10*N+it]*trkErr[9*N+it]+kGain[11*N+it]*trkErr[13*N+it]);
    newErr[17*N+it] = trkErr[17*N+it] - (kGain[ 9*N+it]*trkErr[5*N+it]+kGain[10*N+it]*trkErr[10*N+it]+kGain[11*N+it]*trkErr[14*N+it]);

    newErr[18*N+it] = trkErr[18*N+it] - (kGain[12*N+it]*trkErr[4*N+it]+kGain[13*N+it]*trkErr[9*N+it]+kGain[14*N+it]*trkErr[13*N+it]);
    newErr[19*N+it] = trkErr[19*N+it] - (kGain[12*N+it]*trkErr[5*N+it]+kGain[13*N+it]*trkErr[10*N+it]+kGain[14*N+it]*trkErr[14*N+it]);

    newErr[20*N+it] = trkErr[20*N+it] - (kGain[15*N+it]*trkErr[5*N+it]+kGain[16*N+it]*trkErr[10*N+it]+kGain[17*N+it]*trkErr[14*N+it]);
    
    inPar(iparX, it)     = xnew;
    inPar(iparY, it)     = ynew;
    inPar(iparZ, it)     = znew;
    inPar(iparIpt, it)   = ptnew;
    inPar(iparPhi, it)   = phinew;
    inPar(iparTheta, it) = thetanew;
    
 #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr[ i*N+it] = trkErr[ i*N+it] - newErr[ i*N+it];
    }

  }
  
  return;
}              

template <int N = 1>
void KalmanUpdate_v2(MP6x6SF_<N> &trkErr, MP6F_<N> &inPar, const MP3x3SF_<N> &hitErr, const MP3F_<N> &msP){
   MP2x2SF_<N> resErr_loc;
   MP2x6_<N> kGain;
   MP2F_<N> res_loc;
   MP6x6SF_<N> newErr;

   // AddIntoUpperLeft2x2(psErr, msErr, resErr);
   {
     resErr_loc[0] = trkErr[0] + hitErr[0];
     resErr_loc[1] = trkErr[1] + hitErr[1];
     resErr_loc[2] = trkErr[2] + hitErr[2];
   }

   // Matriplex::InvertCramerSym(resErr);
   {
     const double det = (double)resErr_loc[0] * resErr_loc[2] -
                        (double)resErr_loc[1] * resErr_loc[1];
     const float s   = 1.f / det;
     const float tmp = s * resErr_loc[2];
     resErr_loc[1] *= -s;
     resErr_loc[2]  = s * resErr_loc[0];
     resErr_loc[0]  = tmp;
   }

   // KalmanGain(psErr, resErr, K);
   {
      kGain[ 0] = trkErr[ 0]*resErr_loc[ 0] + trkErr[ 1]*resErr_loc[ 1];
      kGain[ 1] = trkErr[ 0]*resErr_loc[ 1] + trkErr[ 1]*resErr_loc[ 2];
      kGain[ 2] = trkErr[ 1]*resErr_loc[ 0] + trkErr[ 2]*resErr_loc[ 1];
      kGain[ 3] = trkErr[ 1]*resErr_loc[ 1] + trkErr[ 2]*resErr_loc[ 2];
      kGain[ 4] = trkErr[ 3]*resErr_loc[ 0] + trkErr[ 4]*resErr_loc[ 1];
      kGain[ 5] = trkErr[ 3]*resErr_loc[ 1] + trkErr[ 4]*resErr_loc[ 2];
      kGain[ 6] = trkErr[ 6]*resErr_loc[ 0] + trkErr[ 7]*resErr_loc[ 1];
      kGain[ 7] = trkErr[ 6]*resErr_loc[ 1] + trkErr[ 7]*resErr_loc[ 2];
      kGain[ 8] = trkErr[10]*resErr_loc[ 0] + trkErr[11]*resErr_loc[ 1];
      kGain[ 9] = trkErr[10]*resErr_loc[ 1] + trkErr[11]*resErr_loc[ 2];
      kGain[10] = trkErr[15]*resErr_loc[ 0] + trkErr[16]*resErr_loc[ 1];
      kGain[11] = trkErr[15]*resErr_loc[ 1] + trkErr[16]*resErr_loc[ 2];
   }

   // SubtractFirst2(msPar, psPar, res);
   // MultResidualsAdd(K, psPar, res, outPar);
   {
     res_loc[0] =  msP[iparX] - inPar[iparX];
     res_loc[1] =  msP[iparY] - inPar[iparY];

     inPar[iparX] = inPar[iparX] + kGain[ 0] * res_loc[ 0] + kGain[ 1] * res_loc[ 1];
     inPar[iparY] = inPar[iparY] + kGain[ 2] * res_loc[ 0] + kGain[ 3] * res_loc[ 1];
     inPar[iparZ] = inPar[iparZ] + kGain[ 4] * res_loc[ 0] + kGain[ 5] * res_loc[ 1];
     inPar[iparIpt] = inPar[iparIpt] + kGain[ 6] * res_loc[ 0] + kGain[ 7] * res_loc[ 1];
     inPar[iparPhi] = inPar[iparPhi] + kGain[ 8] * res_loc[ 0] + kGain[ 9] * res_loc[ 1];
     inPar[iparTheta] = inPar[iparTheta] + kGain[10] * res_loc[ 0] + kGain[11] * res_loc[ 1];
     //note: if ipt changes sign we should update the charge, or we should get rid of the charge altogether and just use the sign of ipt
   }
   // squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
   // missing

   // KHC(K, psErr, outErr);
   // outErr.Subtract(psErr, outErr);
   {
      newErr[ 0] = kGain[ 0]*trkErr[ 0] + kGain[ 1]*trkErr[ 1];
      newErr[ 1] = kGain[ 2]*trkErr[ 0] + kGain[ 3]*trkErr[ 1];
      newErr[ 2] = kGain[ 2]*trkErr[ 1] + kGain[ 3]*trkErr[ 2];
      newErr[ 3] = kGain[ 4]*trkErr[ 0] + kGain[ 5]*trkErr[ 1];
      newErr[ 4] = kGain[ 4]*trkErr[ 1] + kGain[ 5]*trkErr[ 2];
      newErr[ 5] = kGain[ 4]*trkErr[ 3] + kGain[ 5]*trkErr[ 4];
      newErr[ 6] = kGain[ 6]*trkErr[ 0] + kGain[ 7]*trkErr[ 1];
      newErr[ 7] = kGain[ 6]*trkErr[ 1] + kGain[ 7]*trkErr[ 2];
      newErr[ 8] = kGain[ 6]*trkErr[ 3] + kGain[ 7]*trkErr[ 4];
      newErr[ 9] = kGain[ 6]*trkErr[ 6] + kGain[ 7]*trkErr[ 7];
      newErr[10] = kGain[ 8]*trkErr[ 0] + kGain[ 9]*trkErr[ 1];
      newErr[11] = kGain[ 8]*trkErr[ 1] + kGain[ 9]*trkErr[ 2];
      newErr[12] = kGain[ 8]*trkErr[ 3] + kGain[ 9]*trkErr[ 4];
      newErr[13] = kGain[ 8]*trkErr[ 6] + kGain[ 9]*trkErr[ 7];
      newErr[14] = kGain[ 8]*trkErr[10] + kGain[ 9]*trkErr[11];
      newErr[15] = kGain[10]*trkErr[ 0] + kGain[11]*trkErr[ 1];
      newErr[16] = kGain[10]*trkErr[ 1] + kGain[11]*trkErr[ 2];
      newErr[17] = kGain[10]*trkErr[ 3] + kGain[11]*trkErr[ 4];
      newErr[18] = kGain[10]*trkErr[ 6] + kGain[11]*trkErr[ 7];
      newErr[19] = kGain[10]*trkErr[10] + kGain[11]*trkErr[11];
      newErr[20] = kGain[10]*trkErr[15] + kGain[11]*trkErr[16];

      newErr[ 0] = trkErr[ 0] - newErr[ 0];
      newErr[ 1] = trkErr[ 1] - newErr[ 1];
      newErr[ 2] = trkErr[ 2] - newErr[ 2];
      newErr[ 3] = trkErr[ 3] - newErr[ 3];
      newErr[ 4] = trkErr[ 4] - newErr[ 4];
      newErr[ 5] = trkErr[ 5] - newErr[ 5];
      newErr[ 6] = trkErr[ 6] - newErr[ 6];
      newErr[ 7] = trkErr[ 7] - newErr[ 7];
      newErr[ 8] = trkErr[ 8] - newErr[ 8];
      newErr[ 9] = trkErr[ 9] - newErr[ 9];
      newErr[10] = trkErr[10] - newErr[10];
      newErr[11] = trkErr[11] - newErr[11];
      newErr[12] = trkErr[12] - newErr[12];
      newErr[13] = trkErr[13] - newErr[13];
      newErr[14] = trkErr[14] - newErr[14];
      newErr[15] = trkErr[15] - newErr[15];
      newErr[16] = trkErr[16] - newErr[16];
      newErr[17] = trkErr[17] - newErr[17];
      newErr[18] = trkErr[18] - newErr[18];
      newErr[19] = trkErr[19] - newErr[19];
      newErr[20] = trkErr[20] - newErr[20];
   }

  {
    #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr[ i] = trkErr[ i] - newErr[ i];
    }
  }
}


constexpr auto kfact= 100./(-0.299792458*3.8112);
//constexpr auto kfact= 100/3.8;

template<int N = 1>
void propagateToZ(const MP6x6SF_<N> &inErr, const MP6F_<N> &inPar, const MP1I_<N> &inChg, 
                  const MP3F_<N> &msP, MP6x6SF_<N> &outErr, MP6F_<N> &outPar) {
  
  MP6x6F_<N> errorProp;
  MP6x6F_<N> temp;
//#pragma omp simd
  for (size_t it=0;it<N;++it) {	
    const float zout = msP(iparZ,it);
    //note: in principle charge is not needed and could be the sign of ipt
    const float k = inChg[it]*kfact;
    const float deltaZ = zout - inPar(iparZ,it);
    const float ipt  = inPar(iparIpt,it);
    const float pt   = 1.f/ipt;
    const float phi  = inPar(iparPhi,it);
    const float cosP = cosf(phi);
    const float sinP = sinf(phi);
    const float theta= inPar(iparTheta,it);
    const float cosT = cosf(theta);
    const float sinT = sinf(theta);
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT  = 1.f/cosT;
    const float icosTk = icosT/k;
    const float alpha  = deltaZ*sinT*ipt*icosTk;
    //const float alpha = deltaZ*sinT*ipt(inPar,it)/(cosT*k);
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;
    //
    outPar(iparX, it)     = inPar(iparX,it) + k*(pxin*sina - pyin*(1.f-cosa));
    outPar(iparY, it)     = inPar(iparY,it) + k*(pyin*sina + pxin*(1.f-cosa));
    outPar(iparZ, it)     = zout;
    outPar(iparIpt, it)   = ipt;
    outPar(iparPhi, it)   = phi +alpha;
    outPar(iparTheta, it) = theta;
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
    
    //for (size_t i=0;i<6;++i) errorProp[bsize*PosInMtrx(i,i,6) + it] = 1.;
    errorProp[PosInMtrx(0,0,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6, N) + it] = 1.0f;
    //
    errorProp[PosInMtrx(0,1,6, N) + it] = 0.f;
    errorProp[PosInMtrx(0,2,6, N) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp[PosInMtrx(0,3,6, N) + it] = cosP*sinT*deltaZ*cosa*(1.f-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.f-cCosPsina))*(pt*pt);
    errorProp[PosInMtrx(0,4,6, N) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.f-cCosPsina));
    errorProp[PosInMtrx(0,5,6, N) + it] = cosP*deltaZ*cosa*(1.f-sinP*sCosPsina)*(icosT*icosT);
    errorProp[PosInMtrx(1,2,6, N) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp[PosInMtrx(1,3,6, N) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.f-cCosPsina))*(pt*pt);
    errorProp[PosInMtrx(1,4,6, N) + it] = (k*pt)*(-sinP*(1.f-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp[PosInMtrx(1,5,6, N) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp[PosInMtrx(4,2,6, N) + it] = -ipt*sinT*(icosTk);//!
    errorProp[PosInMtrx(4,3,6, N) + it] = sinT*deltaZ*(icosTk);
    errorProp[PosInMtrx(4,5,6, N) + it] = ipt*deltaZ*(icosT*icosTk);//!
  }
  
  MultHelixPropEndcap<N>(errorProp, inErr, temp);
  MultHelixPropTranspEndcap<N>(errorProp, temp, outErr);
  
  return;
}


int main (int argc, char* argv[]) {

#ifdef include_data
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

   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66[PosInMtrx(0,0,6)]],
                                               inputtrk.cov[SymOffsets66[PosInMtrx(1,1,6)]],
                                           inputtrk.cov[SymOffsets66[PosInMtrx(2,2,6)]]);
   for (size_t lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }


   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;
   //
   srand(1);
   //
   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   //~//create fake objects to emulate data transfers
   std::vector<MPTRK> h_outtrcks(nevts*nb);
   //
   std::vector<MPTRK> h_trcks(nevts*nb);
   prepareTracks(h_trcks, inputtrk);
   // 
   std::vector<MPHIT> h_hits(nlayer*nevts*nb);
   prepareHits(h_hits, inputhits);
   //
   std::vector<MPTRK> outtrcks(nevts*nb);
   //
   std::vector<MPTRK> trcks(nevts*nb);
   //
   std::vector<MPHIT> hits(nlayer*nevts*nb);
   //
   //
   auto policy = std::execution::par_unseq;
   //enforce data migration:
   std::copy(policy, h_outtrcks.begin(), h_outtrcks.end(), outtrcks.begin());
   

   auto p2z_kernels = [=,btracksPtr    = trcks.data(),
                         outtracksPtr  = outtrcks.data(),
                         bhitsPtr      = hits.data()] (auto&& i) {
                         //
                         constexpr int  N             = use_cuda ? 1 : bsize;
                         constexpr int  layers        = nlayer;
                         //
                         const int tid       = use_cuda ? i / bsize : i;
                         const int batch_id  = use_cuda ? i % bsize : 0;
                         //  
                         MPTRK_<N> obtracks;
                          
                         const auto& btracks = btracksPtr[tid].load<N>(batch_id);
                         obtracks = btracks;
                         //
#pragma unroll                         
                         for(int layer=0; layer<nlayer; ++layer) {
                           //
                           const auto& bhits = bhitsPtr[layer+layers*tid].load<N>(batch_id);
                           //
                           propagateToZ<N>(obtracks.cov, obtracks.par, obtracks.q, bhits.pos, obtracks.cov, obtracks.par);
                           //KalmanUpdate<N>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                           KalmanUpdate_v2<N>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                           //
                         }
                         //
                         outtracksPtr[tid].save<N>(obtracks, batch_id);
                       };


   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   double wall_time = 0.0;
 
   const int phys_length     = nevts*nb;
   const int tot_phys_length = phys_length*(use_cuda ? bsize : 1);//re-scale the exe domain for the cuda backend! 
      
   auto outer_loop_range = views::iota(0, tot_phys_length);
   if constexpr (include_data_transfer == false){
#ifdef USE_PARALLEL_COPY
     //enforce data migration:
     std::copy(policy, h_trcks.begin(), h_trcks.end(), trcks.begin());
     std::copy(policy, h_hits.begin(), h_hits.end(), hits.begin());
#else
     //just a regular copy, no migration
     std::copy(h_trcks.begin(), h_trcks.end(), trcks.begin());
     std::copy(h_hits.begin(), h_hits.end(), hits.begin());
#endif
   }  

   //
   auto wall_start = std::chrono::high_resolution_clock::now();
   //
   for(int itr=0; itr<NITER; itr++) {
     if constexpr (include_data_transfer){
#ifdef USE_PARALLEL_COPY
       //enforce data migration:
       std::copy(policy, h_trcks.begin(), h_trcks.end(), trcks.begin());
       std::copy(policy, h_hits.begin(), h_hits.end(), hits.begin());
#else
       //just a regular copy, no migration
       std::copy(h_trcks.begin(), h_trcks.end(), trcks.begin());
       std::copy(h_hits.begin(), h_hits.end(), hits.begin());
#endif
     }  
     std::for_each(policy,
                   begin(outer_loop_range),
                   end(outer_loop_range),
                   p2z_kernels);
                   
     if constexpr (include_data_transfer) {
#ifdef USE_PARALLEL_COPY
        std::copy(policy, outtrcks.begin(), outtrcks.end(), h_outtrcks.begin());
#else
        std::copy(outtrcks.begin(), outtrcks.end(), h_outtrcks.begin());
#endif
     }

   } //end of itr loop
   //
   auto wall_stop = std::chrono::high_resolution_clock::now();
   //

   if constexpr (include_data_transfer == false) {
#ifdef USE_PARALLEL_COPY
      std::copy(policy, outtrcks.begin(), outtrcks.end(), h_outtrcks.begin());
#else
      std::copy(outtrcks.begin(), outtrcks.end(), h_outtrcks.begin());
#endif
   }
   //
   auto wall_diff = wall_stop - wall_start;
   //
   wall_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;

   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);

   auto outtrk = outtrcks.data();
   auto hit    = hits.data();

   int nnans = 0, nfail = 0;
   double avgx = 0, avgy = 0, avgz = 0;
   double avgpt = 0, avgphi = 0, avgtheta = 0;
   double avgdx = 0, avgdy = 0, avgdz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float pt_ = 1./ipt(outtrk,ie,it);
       float phi_ = phi(outtrk,ie,it);
       float theta_ = theta(outtrk,ie,it);
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
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
   avgpt = avgpt/double(nevts*ntrks);
   avgphi = avgphi/double(nevts*ntrks);
   avgtheta = avgtheta/double(nevts*ntrks);
   avgx = avgx/double(nevts*ntrks);
   avgy = avgy/double(nevts*ntrks);
   avgz = avgz/double(nevts*ntrks);
   avgdx = avgdx/double(nevts*ntrks);
   avgdy = avgdy/double(nevts*ntrks);
   avgdz = avgdz/double(nevts*ntrks);

   double stdx = 0, stdy = 0, stdz = 0;
   double stddx = 0, stddy = 0, stddz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
       float pt_ = 1./ipt(outtrk,ie,it);
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
     }
   }

   stdx = sqrtf(stdx/double(nevts*ntrks));
   stdy = sqrtf(stdy/double(nevts*ntrks));
   stdz = sqrtf(stdz/double(nevts*ntrks));
   stddx = sqrtf(stddx/double(nevts*ntrks));
   stddy = sqrtf(stddy/double(nevts*ntrks));
   stddz = sqrtf(stddz/double(nevts*ntrks));

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

   return 0;
}
