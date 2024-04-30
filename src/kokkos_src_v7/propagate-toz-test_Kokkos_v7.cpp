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

#include <Kokkos_Core.hpp>

//#define DUMP_OUTPUT
#define FIXED_RSEED

#ifndef bsize
#define bsize 32
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    (ntrks/bsize)
#ifndef nevts
#define nevts 100
#endif
#define smear 0.0000001

#ifndef NITER
#define NITER 5
#endif
#ifndef nlayer
#define nlayer 20
#endif

#ifndef num_streams
#define num_streams 1
#endif

#ifndef prepin_hostmem
#define prepin_hostmem 0
#endif

#define ExecSpace Kokkos::DefaultExecutionSpace
#define MemSpace ExecSpace::memory_space

#ifdef KOKKOS_ENABLE_CUDA
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_HIP
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_OPENMPTARGET
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_OPENMP
constexpr bool use_gpu = false;
#else
#ifdef KOKKOS_ENABLE_SYCL
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_THREADS
constexpr bool use_gpu = false;
#else
#ifdef KOKKOS_ENABLE_SERIAL
constexpr bool use_gpu = false;
#endif
#endif
#endif
#endif
#endif
#endif
#endif

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

constexpr int iparX     = 0;
constexpr int iparY     = 1;
constexpr int iparZ     = 2;
constexpr int iparIpt   = 3;
constexpr int iparPhi   = 4;
constexpr int iparTheta = 5;

template <typename T, int N, int bSize = 1> 
struct MPNX {
   T data[N*bSize];

   MPNX() = default;
   MPNX(const MPNX<T, N, bSize> &) = default;
   MPNX(MPNX<T, N, bSize> &&)      = default;

   //basic accessors   
   constexpr T &operator[](int i) { return data[i]; }
   constexpr const T &operator[](int i) const { return data[i]; }
   constexpr T& operator()(const int i, const int j) {return data[i*bSize+j];}
   constexpr const T& operator()(const int i, const int j) const {return data[i*bSize+j];}
   constexpr int size() const { return N*bSize; }    
   //   
   
   KOKKOS_INLINE_FUNCTION  void load(MPNX<T, N, 1>& dst, const int b) const {
#pragma unroll
     for (int ip=0;ip<N;++ip) { //block load    
        dst.data[ip] = data[ip*bSize + b];  
     }    
     
     return;
   }

   KOKKOS_INLINE_FUNCTION  void save(const MPNX<T, N, 1>& src, const int b) { 
#pragma unroll
     for (int ip=0;ip<N;++ip) {     
         data[ip*bSize + b] = src.data[ip]; 
     }    
     
     return;
   }    
   
   auto operator=(const MPNX&) -> MPNX& = default;
   auto operator=(MPNX&&     ) -> MPNX& = default; 
};

// external data formats:
using MP1I    = MPNX<int,   1 , bsize>;
using MP22I   = MPNX<int,   22, bsize>;
using MP2F    = MPNX<float, 2 , bsize>;
using MP3F    = MPNX<float, 3 , bsize>;
using MP6F    = MPNX<float, 6 , bsize>;
using MP2x2SF = MPNX<float, 3 , bsize>;
using MP3x3SF = MPNX<float, 6 , bsize>;
using MP6x6SF = MPNX<float, 21, bsize>;
using MP6x6F  = MPNX<float, 36, bsize>;
using MP3x3   = MPNX<float, 9 , bsize>;
using MP3x6   = MPNX<float, 18, bsize>;
using MP2x6   = MPNX<float, 12, bsize>;

// internal data formats:
template<int bSize=1> using MP1I_    = MPNX<int,   1 ,bSize>;
template<int bSize=1> using MP22I_   = MPNX<int,   22,bSize>;
template<int bSize=1> using MP2F_    = MPNX<float, 2 ,bSize>;
template<int bSize=1> using MP3F_    = MPNX<float, 3 ,bSize>;
template<int bSize=1> using MP6F_    = MPNX<float, 6 ,bSize>;
template<int bSize=1> using MP2x2SF_ = MPNX<float, 3 ,bSize>;
template<int bSize=1> using MP3x3SF_ = MPNX<float, 6 ,bSize>;
template<int bSize=1> using MP6x6SF_ = MPNX<float, 21,bSize>;
template<int bSize=1> using MP6x6F_  = MPNX<float, 36,bSize>;
template<int bSize=1> using MP3x3_   = MPNX<float, 9 ,bSize>;
template<int bSize=1> using MP3x6_   = MPNX<float, 18,bSize>;
template<int bSize=1> using MP2x6_   = MPNX<float, 12,bSize>;

template<int N=1>
struct MPTRK_ {
  MP6F_<N>    par;
  MP6x6SF_<N> cov;
  MP1I_<N>    q;
};

template<int N=1>
struct MPHIT_ {
  MP3F_<N>    pos;
  MP3x3SF_<N> cov;
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  template<int S>
  KOKKOS_INLINE_FUNCTION  const auto load_component (const int batch_id) const{//b is a batch idx

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
  KOKKOS_INLINE_FUNCTION   void save_component(MPTRK_<S> &src, const int batch_id) {

    if constexpr (std::is_same<MP6F, MP6F_<S>>::value
                  and std::is_same<MP6x6SF, MP6x6SF_<S>>::value
                  and std::is_same<MP1I, MP1I_<S>>::value) { //just do a copy of the whole objects

      this->par = src.par;
      this->cov = src.cov;
      this->q   = src.q;

    } else{
    this->par.save(src.par, batch_id);
    this->cov.save(src.cov, batch_id);
    this->q.save(src.q, batch_id);
    }

    return;
  }
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
  //
  template<int S>
  KOKKOS_INLINE_FUNCTION   const auto load_component(const int batch_id) const {
    MPHIT_<S> dst;

    if constexpr (std::is_same<MP3F, MP3F_<S>>::value
                  and std::is_same<MP3x3SF, MP3x3SF_<S>>::value) { //just do a copy of the whole object
      dst.pos = this->pos;
      dst.cov = this->cov;
    } else { //ok, do manual load of the batch component instead
    this->pos.load(dst.pos, batch_id);
    this->cov.load(dst.cov, batch_id);
    }
    return dst;
  }
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

KOKKOS_FUNCTION MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

KOKKOS_FUNCTION const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

KOKKOS_FUNCTION MPTRK* bTk(const Kokkos::View<MPTRK*> &tracks, size_t ev, size_t ib) {
  return &(tracks(ib + nb*ev));
}

KOKKOS_FUNCTION int q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
KOKKOS_FUNCTION float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
KOKKOS_FUNCTION float x    (const MP6F* bpars, size_t it){ return par(bpars, it, 0); }
KOKKOS_FUNCTION float y    (const MP6F* bpars, size_t it){ return par(bpars, it, 1); }
KOKKOS_FUNCTION float z    (const MP6F* bpars, size_t it){ return par(bpars, it, 2); }
KOKKOS_FUNCTION float ipt  (const MP6F* bpars, size_t it){ return par(bpars, it, 3); }
KOKKOS_FUNCTION float phi  (const MP6F* bpars, size_t it){ return par(bpars, it, 4); }
KOKKOS_FUNCTION float theta(const MP6F* bpars, size_t it){ return par(bpars, it, 5); }
//
KOKKOS_FUNCTION float par(const MPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
KOKKOS_FUNCTION float x    (const MPTRK* btracks, size_t it){ return par(btracks, it, 0); }
KOKKOS_FUNCTION float y    (const MPTRK* btracks, size_t it){ return par(btracks, it, 1); }
KOKKOS_FUNCTION float z    (const MPTRK* btracks, size_t it){ return par(btracks, it, 2); }
KOKKOS_FUNCTION float ipt  (const MPTRK* btracks, size_t it){ return par(btracks, it, 3); }
KOKKOS_FUNCTION float phi  (const MPTRK* btracks, size_t it){ return par(btracks, it, 4); }
KOKKOS_FUNCTION float theta(const MPTRK* btracks, size_t it){ return par(btracks, it, 5); }
//
KOKKOS_FUNCTION float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
KOKKOS_FUNCTION float par(const Kokkos::View<MPTRK*> &tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
KOKKOS_FUNCTION float x    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
KOKKOS_FUNCTION float y    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
KOKKOS_FUNCTION float z    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
KOKKOS_FUNCTION float ipt  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
KOKKOS_FUNCTION float phi  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
KOKKOS_FUNCTION float theta(const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
KOKKOS_FUNCTION void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
KOKKOS_FUNCTION void setx    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 0, val); }
KOKKOS_FUNCTION void sety    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 1, val); }
KOKKOS_FUNCTION void setz    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 2, val); }
KOKKOS_FUNCTION void setipt  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 3, val); }
KOKKOS_FUNCTION void setphi  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 4, val); }
KOKKOS_FUNCTION void settheta(MP6F* bpars, size_t it, float val){ setpar(bpars, it, 5, val); }
//
KOKKOS_FUNCTION void setpar(MPTRK* btracks, size_t it, size_t ipar, float val){
  setpar(&(*btracks).par,it,ipar,val);
}
KOKKOS_FUNCTION void setx    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 0, val); }
KOKKOS_FUNCTION void sety    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 1, val); }
KOKKOS_FUNCTION void setz    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 2, val); }
KOKKOS_FUNCTION void setipt  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 3, val); }
KOKKOS_FUNCTION void setphi  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 4, val); }
KOKKOS_FUNCTION void settheta(MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 5, val); }

KOKKOS_FUNCTION const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
KOKKOS_FUNCTION const MPHIT* bHit(const Kokkos::View<MPHIT*> &hits, size_t ev, size_t ib) {
  return &(hits(ib + nb*ev));
}
KOKKOS_FUNCTION const MPHIT* bHit(const MPHIT* &hits, size_t ev, size_t ib,size_t lay) {
  return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
KOKKOS_FUNCTION const MPHIT* bHit(const Kokkos::View<MPHIT*> &hits, size_t ev, size_t ib,size_t lay) {
  return &(hits(lay + (ib*nlayer) +(ev*nlayer*nb)));
}
//
KOKKOS_FUNCTION float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
KOKKOS_FUNCTION float x(const MP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
KOKKOS_FUNCTION float y(const MP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
KOKKOS_FUNCTION float z(const MP3F* hpos, size_t it)    { return pos(hpos, it, 2); }
//
KOKKOS_FUNCTION float pos(const MPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
KOKKOS_FUNCTION float x(const MPHIT* hits, size_t it)    { return pos(hits, it, 0); }
KOKKOS_FUNCTION float y(const MPHIT* hits, size_t it)    { return pos(hits, it, 1); }
KOKKOS_FUNCTION float z(const MPHIT* hits, size_t it)    { return pos(hits, it, 2); }
//
KOKKOS_FUNCTION float pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  //[DEBUG on Dec. 22, 2020] add 4th argument(nlayer-1) to bHit() below.
  const MPHIT* bhits = bHit(hits, ev, ib,nlayer-1);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
KOKKOS_FUNCTION float x(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
KOKKOS_FUNCTION float y(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
KOKKOS_FUNCTION float z(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }

KOKKOS_FUNCTION float pos(const Kokkos::View<MPHIT*> &hits, size_t it, size_t ipar){
  return pos(&(hits(0).pos),it,ipar);
}
KOKKOS_FUNCTION float x(const Kokkos::View<MPHIT*> &hits, size_t it)    { return pos(hits, it, 0); }
KOKKOS_FUNCTION float y(const Kokkos::View<MPHIT*> &hits, size_t it)    { return pos(hits, it, 1); }
KOKKOS_FUNCTION float z(const Kokkos::View<MPHIT*> &hits, size_t it)    { return pos(hits, it, 2); }
//
KOKKOS_FUNCTION float pos(const Kokkos::View<MPHIT*> &hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  //[DEBUG by Seyong on Dec. 28, 2020] add 4th argument(nlayer-1) to bHit() below.
  const MPHIT* bhits = bHit(hits, ev, ib, nlayer-1);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
KOKKOS_FUNCTION float x(const Kokkos::View<MPHIT*> &hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
KOKKOS_FUNCTION float y(const Kokkos::View<MPHIT*> &hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
KOKKOS_FUNCTION float z(const Kokkos::View<MPHIT*> &hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }

#if prepin_hostmem == 1
void prepareTracks(ATRK inputtrk, Kokkos::View<MPTRK*, ExecSpace::array_layout, Kokkos::CudaHostPinnedSpace> &result) {
#else
void prepareTracks(ATRK inputtrk, Kokkos::View<MPTRK*>::HostMirror &result) {
#endif
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
    	//par
    	for (size_t ip=0;ip<6;++ip) {
    	  result(ib + nb*ie).par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
    	}
    	//cov, scale by factor 100
    	for (size_t ip=0;ip<21;++ip) {
    	  result(ib + nb*ie).cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
    	}
    	//q
    	result(ib + nb*ie).q.data[it] = inputtrk.q;//can't really smear this or fit will be wrong
      }
    }
  }
}

#if prepin_hostmem == 1
void prepareHits(AHIT* inputhits, Kokkos::View<MPHIT*, ExecSpace::array_layout, Kokkos::CudaHostPinnedSpace> &result) {
#else
void prepareHits(AHIT* inputhits, Kokkos::View<MPHIT*>::HostMirror &result) {
#endif
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {

    struct AHIT inputhit = inputhits[lay];

    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  result(lay+nlayer*(ib + nb*ie)).pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  result(lay+nlayer*(ib + nb*ie)).cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
}

template<size_t N=1>
KOKKOS_FUNCTION void MultHelixPropEndcap(const MP6x6F_<N> &a, const MP6x6SF_<N> &b, MP6x6F_<N> &c) {
  //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, bsize), [&] (const size_t n)
  #pragma omp simd
  for (int n =0;n<N;n++)
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
  //});
}

template<size_t N=1>
KOKKOS_FUNCTION void MultHelixPropTranspEndcap(const MP6x6F_<N> &a, const MP6x6F_<N> &b, MP6x6SF_<N> &c) {
  //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, bsize), [&] (const size_t n)
  #pragma omp simd
  for (int n =0;n<N;n++)
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
  //});
}

template<size_t N=1>
KOKKOS_FUNCTION void KalmanUpdate_v2(MP6x6SF_<N> &trkErr_, MP6F_<N> &inPar_, const MP3x3SF_<N> &hitErr_, const MP3F_<N> &msP_){

    MP2x2SF_<N> resErr_loc;
    MP2x6_<N> kGain;
    MP2F_<N> res_loc;
    MP6x6SF_<N> newErr;
  // printf("kalman in: x=%7f, y=%7f, z=%7f, ipt=%7f, phi=%7f, theta=%7f \n", x    (inPar, 0), y    (inPar, 0), z    (inPar, 0), ipt  (inPar, 0), phi  (inPar, 0), theta(inPar, 0));
  // printf("oldErr ");
  // for (int i = 0; i < 21; i++){
  //   printf("%10f \n",trkErr_[ i*N+0]);
  // }
  // //printf("\n");

   // AddIntoUpperLeft2x2(psErr, msErr, resErr);
   //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N),[&](const size_t it) 
   #pragma omp simd
   for (int it =0;it<N;it++)
   {
     resErr_loc.data[0*N+it] = trkErr_[0*N+it] + hitErr_[0*N+it];
     resErr_loc.data[1*N+it] = trkErr_[1*N+it] + hitErr_[1*N+it];
     resErr_loc.data[2*N+it] = trkErr_[2*N+it] + hitErr_[2*N+it];
   }
   //});

   // Matriplex::InvertCramerSym(resErr);
   //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N),[&](const size_t it) 
   #pragma omp simd
   for (int it =0;it<N;it++)
   {
     const double det = (double)resErr_loc.data[0*N+it] * resErr_loc.data[2*N+it] -
                        (double)resErr_loc.data[1*N+it] * resErr_loc.data[1*N+it];
     const float s   = 1.f / det;
     const float tmp = s * resErr_loc.data[2*N+it];
     resErr_loc.data[1*N+it] *= -s;
     resErr_loc.data[2*N+it]  = s * resErr_loc.data[0*N+it];
     resErr_loc.data[0*N+it]  = tmp;
   }
   //});

  //    printf("resErrLoc ");
  // for (int i = 0; i < 3; i++){
  //   printf("%10f \n",resErr_loc.data[ i*N+0]);
  // }

   // KalmanGain(psErr, resErr, K);
   //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N),[&](const size_t it) 
   #pragma omp simd
   for (int it =0;it<N;it++)
   {
      kGain.data[ 0*N+it] = trkErr_[ 0*N+it]*resErr_loc.data[ 0*N+it] + trkErr_[ 1*N+it]*resErr_loc.data[ 1*N+it];
      kGain.data[ 1*N+it] = trkErr_[ 0*N+it]*resErr_loc.data[ 1*N+it] + trkErr_[ 1*N+it]*resErr_loc.data[ 2*N+it];
      kGain.data[ 2*N+it] = trkErr_[ 1*N+it]*resErr_loc.data[ 0*N+it] + trkErr_[ 2*N+it]*resErr_loc.data[ 1*N+it];
      kGain.data[ 3*N+it] = trkErr_[ 1*N+it]*resErr_loc.data[ 1*N+it] + trkErr_[ 2*N+it]*resErr_loc.data[ 2*N+it];
      kGain.data[ 4*N+it] = trkErr_[ 3*N+it]*resErr_loc.data[ 0*N+it] + trkErr_[ 4*N+it]*resErr_loc.data[ 1*N+it];
      kGain.data[ 5*N+it] = trkErr_[ 3*N+it]*resErr_loc.data[ 1*N+it] + trkErr_[ 4*N+it]*resErr_loc.data[ 2*N+it];
      kGain.data[ 6*N+it] = trkErr_[ 6*N+it]*resErr_loc.data[ 0*N+it] + trkErr_[ 7*N+it]*resErr_loc.data[ 1*N+it];
      kGain.data[ 7*N+it] = trkErr_[ 6*N+it]*resErr_loc.data[ 1*N+it] + trkErr_[ 7*N+it]*resErr_loc.data[ 2*N+it];
      kGain.data[ 8*N+it] = trkErr_[10*N+it]*resErr_loc.data[ 0*N+it] + trkErr_[11*N+it]*resErr_loc.data[ 1*N+it];
      kGain.data[ 9*N+it] = trkErr_[10*N+it]*resErr_loc.data[ 1*N+it] + trkErr_[11*N+it]*resErr_loc.data[ 2*N+it];
      kGain.data[10*N+it] = trkErr_[15*N+it]*resErr_loc.data[ 0*N+it] + trkErr_[16*N+it]*resErr_loc.data[ 1*N+it];
      kGain.data[11*N+it] = trkErr_[15*N+it]*resErr_loc.data[ 1*N+it] + trkErr_[16*N+it]*resErr_loc.data[ 2*N+it];
   }
   //});

  // printf("kGain ");
  // for (int i = 0; i < 12; i++){
  //   printf("%10f \n",kGain.data[ i*N+0]);
  // }
  // //printf("\n");

  // SubtractFirst2(msPar, psPar, res);
   // MultResidualsAdd(K, psPar, res, outPar);
   //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N),[&](const size_t it) 
   #pragma omp simd
   for (int it =0;it<N;it++)
   {
     res_loc.data[0*N+it] =  msP_(iparX,it) - inPar_(iparX,it);
     res_loc.data[1*N+it] =  msP_(iparY,it) - inPar_(iparY,it);

     inPar_(iparX, it) = inPar_(iparX, it) + kGain.data[ 0*N+it] * res_loc.data[ 0*N+it] + kGain.data[ 1*N+it] * res_loc.data[ 1*N+it];
     inPar_(iparY, it) = inPar_(iparY, it) + kGain.data[ 2*N+it] * res_loc.data[ 0*N+it] + kGain.data[ 3*N+it] * res_loc.data[ 1*N+it];
     inPar_(iparZ, it) = inPar_(iparZ, it) + kGain.data[ 4*N+it] * res_loc.data[ 0*N+it] + kGain.data[ 5*N+it] * res_loc.data[ 1*N+it];
     inPar_(iparIpt, it) = inPar_(iparIpt, it) + kGain.data[ 6*N+it] * res_loc.data[ 0*N+it] + kGain.data[ 7*N+it] * res_loc.data[ 1*N+it];
     inPar_(iparPhi, it) = inPar_(iparPhi, it) + kGain.data[ 8*N+it] * res_loc.data[ 0*N+it] + kGain.data[ 9*N+it] * res_loc.data[ 1*N+it];
     inPar_(iparTheta, it) = inPar_(iparTheta, it) + kGain.data[10*N+it] * res_loc.data[ 0*N+it] + kGain.data[11*N+it] * res_loc.data[ 1*N+it];
     //note: if ipt changes sign we should update the charge, or we should get rid of the charge altogether and just use the sign of ipt
   }
   //});

   // printf("kalman out: x=%7f, y=%7f, z=%7f, ipt=%7f, phi=%7f, theta=%7f \n", x    (inPar, 0), y    (inPar, 0), z    (inPar, 0), ipt  (inPar, 0), phi  (inPar, 0), theta(inPar, 0));

   // squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
   // missing

  //  printf("trkErr ");
  // for (int i = 0; i < 21; i++){
  //   printf("%10f ",trkErr_[ i*N+0]);
  // }
  // printf("\n");

   // KHC(K, psErr, outErr);
   // outErr.Subtract(psErr, outErr);
   //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N),[&](const size_t it) 
   #pragma omp simd
   for (int it =0;it<N;it++)
   {
      newErr.data[ 0*N+it] = kGain.data[ 0*N+it]*trkErr_[ 0*N+it] + kGain.data[ 1*N+it]*trkErr_[ 1*N+it];
      newErr.data[ 1*N+it] = kGain.data[ 2*N+it]*trkErr_[ 0*N+it] + kGain.data[ 3*N+it]*trkErr_[ 1*N+it];
      newErr.data[ 2*N+it] = kGain.data[ 2*N+it]*trkErr_[ 1*N+it] + kGain.data[ 3*N+it]*trkErr_[ 2*N+it];
      newErr.data[ 3*N+it] = kGain.data[ 4*N+it]*trkErr_[ 0*N+it] + kGain.data[ 5*N+it]*trkErr_[ 1*N+it];
      newErr.data[ 4*N+it] = kGain.data[ 4*N+it]*trkErr_[ 1*N+it] + kGain.data[ 5*N+it]*trkErr_[ 2*N+it];
      newErr.data[ 5*N+it] = kGain.data[ 4*N+it]*trkErr_[ 3*N+it] + kGain.data[ 5*N+it]*trkErr_[ 4*N+it];
      newErr.data[ 6*N+it] = kGain.data[ 6*N+it]*trkErr_[ 0*N+it] + kGain.data[ 7*N+it]*trkErr_[ 1*N+it];
      newErr.data[ 7*N+it] = kGain.data[ 6*N+it]*trkErr_[ 1*N+it] + kGain.data[ 7*N+it]*trkErr_[ 2*N+it];
      newErr.data[ 8*N+it] = kGain.data[ 6*N+it]*trkErr_[ 3*N+it] + kGain.data[ 7*N+it]*trkErr_[ 4*N+it];
      newErr.data[ 9*N+it] = kGain.data[ 6*N+it]*trkErr_[ 6*N+it] + kGain.data[ 7*N+it]*trkErr_[ 7*N+it];
      newErr.data[10*N+it] = kGain.data[ 8*N+it]*trkErr_[ 0*N+it] + kGain.data[ 9*N+it]*trkErr_[ 1*N+it];
      newErr.data[11*N+it] = kGain.data[ 8*N+it]*trkErr_[ 1*N+it] + kGain.data[ 9*N+it]*trkErr_[ 2*N+it];
      newErr.data[12*N+it] = kGain.data[ 8*N+it]*trkErr_[ 3*N+it] + kGain.data[ 9*N+it]*trkErr_[ 4*N+it];
      newErr.data[13*N+it] = kGain.data[ 8*N+it]*trkErr_[ 6*N+it] + kGain.data[ 9*N+it]*trkErr_[ 7*N+it];
      newErr.data[14*N+it] = kGain.data[ 8*N+it]*trkErr_[10*N+it] + kGain.data[ 9*N+it]*trkErr_[11*N+it];
      newErr.data[15*N+it] = kGain.data[10*N+it]*trkErr_[ 0*N+it] + kGain.data[11*N+it]*trkErr_[ 1*N+it];
      newErr.data[16*N+it] = kGain.data[10*N+it]*trkErr_[ 1*N+it] + kGain.data[11*N+it]*trkErr_[ 2*N+it];
      newErr.data[17*N+it] = kGain.data[10*N+it]*trkErr_[ 3*N+it] + kGain.data[11*N+it]*trkErr_[ 4*N+it];
      newErr.data[18*N+it] = kGain.data[10*N+it]*trkErr_[ 6*N+it] + kGain.data[11*N+it]*trkErr_[ 7*N+it];
      newErr.data[19*N+it] = kGain.data[10*N+it]*trkErr_[10*N+it] + kGain.data[11*N+it]*trkErr_[11*N+it];
      newErr.data[20*N+it] = kGain.data[10*N+it]*trkErr_[15*N+it] + kGain.data[11*N+it]*trkErr_[16*N+it];

  // printf("tmp newErr ");
  // for (int i = 0; i < 21; i++){
  //   printf("%10f ",newErr.data[ i*N+it]);
  // }
  // printf("\n");      
  }
   //});

  //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N),[&](const size_t it) 
  #pragma omp simd
  for (int it =0;it<N;it++)
  {
    #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr_[ i*N+it] = trkErr_[ i*N+it] - newErr.data[ i*N+it];
    }
  }
  //});
  // printf("newErr ");
  // for (int i = 0; i < 21; i++){
  //   printf("%10f ",trkErr_[ i*N+0]);
  // }
  // printf("\n");
}

constexpr float kfact = 100./(-0.299792458*3.8112);

template <int N = 1>
KOKKOS_FUNCTION void propagateToZ(const MP6x6SF_<N> &inErr_, const MP6F_<N> &inPar_,
		  const MP1I_<N> &inChg_, const MP3F_<N> &msP_,
	                MP6x6SF_<N> &outErr_, MP6F_<N> &outPar_) {
  MP6x6F_<N> errorProp;
  MP6x6F_<N> temp;
  auto PosInMtrx = [=](const size_t &&i, const size_t &&j, const size_t &&D, const size_t block_size = 1) constexpr {return block_size*(i*D+j);};
  //
  //Kokkos::parallel_for( Kokkos::TeamVectorRange(teamMember, N), [&](const size_t it) {
    #pragma omp simd
    for( size_t it=0; it<N; it++ ) {
    const float zout = msP_(iparZ,it);
    const float k = inChg_[it]*kfact;//100/3.8;
    const float deltaZ = zout - inPar_(iparZ,it);
    const float pt = 1.0f/inPar_(iparIpt,it);
    const float cosP = cosf(inPar_(iparPhi,it));
    const float sinP = sinf(inPar_(iparPhi,it));
    const float cosT = cosf(inPar_(iparTheta,it));
    const float sinT = sinf(inPar_(iparTheta,it));
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0f/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*inPar_(iparIpt,it)*icosTk;
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;
    outPar_(iparX,it) = inPar_(iparX,it) + k*(pxin*sina - pyin*(1.0f-cosa));
    outPar_(iparY,it) = inPar_(iparY,it) + k*(pyin*sina + pxin*(1.0f-cosa));
    outPar_(iparZ,it) = zout;
    outPar_(iparIpt,it) = inPar_(iparIpt,it);
    outPar_(iparPhi,it) = inPar_(iparPhi,it)+alpha;
    outPar_(iparTheta,it) = inPar_(iparTheta,it);
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);

    //for (size_t i=0;i<6;++i) errorProp[PosInMtrx(i,i,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(0,0,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6,N) + it] = 1.0f;
    //[Dec. 21, 2022] Added to have the same pattern as the cudauvm version.
    errorProp[PosInMtrx(0,1,6,N) + it] = 0.0f;
    errorProp[PosInMtrx(0,2,6,N) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp[PosInMtrx(0,3,6,N) + it] = cosP*sinT*deltaZ*cosa*(1.0f-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.0f-cCosPsina))*(pt*pt);
    errorProp[PosInMtrx(0,4,6,N) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.0f-cCosPsina));
    errorProp[PosInMtrx(0,5,6,N) + it] = cosP*deltaZ*cosa*(1.0f-sinP*sCosPsina)*(icosT*icosT);
    errorProp[PosInMtrx(1,2,6,N) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp[PosInMtrx(1,3,6,N) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.0f-cCosPsina))*(pt*pt);
    errorProp[PosInMtrx(1,4,6,N) + it] = (k*pt)*(-sinP*(1.0f-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp[PosInMtrx(1,5,6,N) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp[PosInMtrx(4,2,6,N) + it] = -inPar_(iparIpt,it)*sinT*(icosTk);
    errorProp[PosInMtrx(4,3,6,N) + it] = sinT*deltaZ*(icosTk);
    errorProp[PosInMtrx(4,5,6,N) + it] = inPar_(iparIpt,it)*deltaZ*(icosT*icosTk);
    }
  //});
  //
  MultHelixPropEndcap<N>(errorProp, inErr_, temp);
  MultHelixPropTranspEndcap<N>(errorProp, temp, outErr_);
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

   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66(0)],
                                               inputtrk.cov[SymOffsets66(1*6+1)],
	                                       inputtrk.cov[SymOffsets66(2*6+2)]);
   for (size_t lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }

   printf("produce nevts=%i ntrks=%i smearing by=%2.1e \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);
   
   long setup_start, setup_stop;
   double setup_time;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
#ifdef FIXED_RSEED
   //[DEBUG by Seyong on Dec. 28, 2020] add an explicit srand(1) call to generate fixed inputs for better debugging.
   srand(1);
#endif
   Kokkos::initialize(argc, argv);
   {


   printf("After kokkos::init\n");
   ExecSpace e;
   e.print_configuration(std::cout, true);

   Kokkos::View<MPTRK*> trk("trk", nevts*nb);
#if prepin_hostmem == 1
   Kokkos::View<MPTRK*, ExecSpace::array_layout, Kokkos::CudaHostPinnedSpace> h_trk("h_trk", nevts*nb);
   prepareTracks(inputtrk, h_trk);
#else
   Kokkos::View<MPTRK*>::HostMirror h_trk = Kokkos::create_mirror_view(trk);
   prepareTracks(inputtrk, h_trk);
#endif
   //Kokkos::deep_copy(trk, h_trk);
 
   Kokkos::View<MPHIT*> hit("hit", nevts*nb*nlayer);
#if prepin_hostmem == 1
   Kokkos::View<MPHIT*, ExecSpace::array_layout, Kokkos::CudaHostPinnedSpace> h_hit("h_hit", nevts*nb*nlayer);
   prepareHits(inputhits, h_hit);
#else
   Kokkos::View<MPHIT*>::HostMirror h_hit = Kokkos::create_mirror_view(hit);
   prepareHits(inputhits, h_hit);
#endif
   //Kokkos::deep_copy(hit, h_hit);

   Kokkos::View<MPTRK*> outtrk("outtrk", nevts*nb);
#if prepin_hostmem == 1
   Kokkos::View<MPTRK*, ExecSpace::array_layout, Kokkos::CudaHostPinnedSpace> h_outtrk("h_outtrk", nevts*nb);
#else
   Kokkos::View<MPTRK*>::HostMirror h_outtrk = Kokkos::create_mirror_view(outtrk);
#endif

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   setup_time = ((double)(setup_stop - setup_start))*0.001;

   printf("done preparing!\n");
   
   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nlayer*nevts*nb*sizeof(struct MPHIT));

   typedef Kokkos::TeamPolicy<>               team_policy;
   typedef Kokkos::TeamPolicy<>::member_type  member_type;

   int team_policy_range = nevts*nb;  // number of teams
   int team_size = use_gpu ? bsize : 1;  // team size
   constexpr int vector_size = use_gpu ? 1 : bsize;  // thread size

#ifndef include_data
   Kokkos::deep_copy(trk, h_trk);
   Kokkos::deep_copy(hit, h_hit);
   Kokkos::fence();
#endif

   auto wall_start = std::chrono::high_resolution_clock::now();

   int itr;
   for(itr=0; itr<NITER; itr++) {
#ifdef include_data
     Kokkos::deep_copy(trk, h_trk);
     Kokkos::deep_copy(hit, h_hit);
#endif
     {
     Kokkos::parallel_for("Kernel", team_policy(team_policy_range,team_size,vector_size),
                                    KOKKOS_LAMBDA( const member_type &teamMember){
		Kokkos::parallel_for( Kokkos::TeamThreadRange(teamMember, teamMember.team_size()), [&](const size_t it) {
         int i = teamMember.league_rank () * teamMember.team_size () + it;
         constexpr int  N             = use_gpu ? 1 : bsize;
         const int tid        = use_gpu ? i / bsize : i;
         const int batch_id   = use_gpu ? i % bsize : 0;
         const MPTRK* btracks_ = trk.data();
         MPTRK* obtracks_ = outtrk.data();
         const MPHIT* bhits_ = hit.data();
         MPTRK_<N> obtracks;
         const auto& btracks = btracks_[tid].load_component<N>(batch_id);
         obtracks = btracks;
         #pragma unroll
         for(size_t layer=0; layer<nlayer; ++layer) {
            const auto& bhits = bhits_[layer+nlayer*tid].load_component<N>(batch_id);
            propagateToZ<N>(obtracks.cov, obtracks.par, obtracks.q, bhits.pos, obtracks.cov, obtracks.par); // vectorized function
            KalmanUpdate_v2<N>(obtracks.cov,obtracks.par,bhits.cov,bhits.pos);
         }
         obtracks_[tid].save_component<N>(obtracks, batch_id);
		}); 
     }); 
   }  
#ifdef include_data
   Kokkos::deep_copy(h_outtrk, outtrk);
#endif
   } //end of itr loop

   //Syncthreads
   Kokkos::fence();
   auto wall_stop = std::chrono::high_resolution_clock::now();
#ifndef include_data
   Kokkos::deep_copy(h_outtrk, outtrk);
   Kokkos::fence();
#endif

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", setup_time);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, setup_time, -1);
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
       double x_ = x(h_outtrk.data(),ie,it);
       double y_ = y(h_outtrk.data(),ie,it);
       double z_ = z(h_outtrk.data(),ie,it);
       double pt_ = 1./ipt(h_outtrk.data(),ie,it);
       double phi_ = phi(h_outtrk.data(),ie,it);
       double theta_ = theta(h_outtrk.data(),ie,it);
       double hx_ = x(h_hit.data(),ie,it);
       double hy_ = y(h_hit.data(),ie,it);
       double hz_ = z(h_hit.data(),ie,it);
       double hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
	   std::isfinite(y_)==false ||
	   std::isfinite(z_)==false ||
	   std::isfinite(pt_)==false ||
	   std::isfinite(phi_)==false ||
	   std::isfinite(theta_)==false
	   ) {
	 nnans++;
	 continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
           fabs( (y_-hy_)/hy_ )>1. ||
           fabs( (z_-hz_)/hz_ )>1. ||
           fabs( (pt_-12.)/12.)>1. ||
           fabs( (phi_-1.3)/1.3)>1. ||
           fabs( (theta_-2.8)/2.8)>1.
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
       double x_ = x(h_outtrk.data(),ie,it);
       double y_ = y(h_outtrk.data(),ie,it);
       double z_ = z(h_outtrk.data(),ie,it);
       double pt_ = 1./ipt(h_outtrk.data(),ie,it);
       double phi_ = phi(h_outtrk.data(),ie,it);
       double theta_ = theta(h_outtrk.data(),ie,it);
       double hx_ = x(h_hit.data(),ie,it);
       double hy_ = y(h_hit.data(),ie,it);
       double hz_ = z(h_hit.data(),ie,it);
       double hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
	   std::isfinite(y_)==false ||
	   std::isfinite(z_)==false ||
	   std::isfinite(pt_)==false ||
	   std::isfinite(phi_)==false ||
	   std::isfinite(theta_)==false
	   ) {
	 continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
           fabs( (y_-hy_)/hy_ )>1. ||
           fabs( (z_-hz_)/hz_ )>1. ||
           fabs( (pt_-12.)/12.)>1. ||
           fabs( (phi_-1.3)/1.3)>1. ||
           fabs( (theta_-2.8)/2.8)>1.
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
       x_ = x(h_outtrk.data(),ie,it);
       y_ = y(h_outtrk.data(),ie,it);
       z_ = z(h_outtrk.data(),ie,it);
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

   }
   Kokkos::finalize();

   return 0;
}
