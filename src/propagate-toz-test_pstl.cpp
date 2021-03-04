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

//BACKEND selector
#if defined(__NVCOMPILER_CUDA__)

#include <thrust/iterator/counting_iterator.h>
using namespace thrust;

#else

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include <tbb/tbb.h>
using namespace tbb;

constexpr int alloc_align  = (2*1024*1024);

#endif //BACKEND selector

   template<typename Tp>
   struct AlignedAllocator {
     public:

       typedef Tp value_type;

       AlignedAllocator () {};

       AlignedAllocator(const AlignedAllocator&) { }

       template<typename Tp1> constexpr AlignedAllocator(const AlignedAllocator<Tp1>&) { }

       ~AlignedAllocator() { }

       Tp* address(Tp& x) const { return &x; }

       std::size_t  max_size() const throw() { return size_t(-1) / sizeof(Tp); }

       [[nodiscard]] Tp* allocate(std::size_t n){

         Tp* ptr = nullptr;
#ifdef __NVCOMPILER_CUDA__
         auto err = cudaMallocManaged((void **)&ptr,n*sizeof(Tp));

         if( err != cudaSuccess ) {
           ptr = (Tp *) NULL;
           std::cerr << " cudaMallocManaged failed for " << n*sizeof(Tp) << " bytes " <<cudaGetErrorString(err)<< std::endl;
           assert(0);
         }
#elif !defined(DPCPP_BACKEND)
         //ptr = (Tp*)aligned_malloc(alloc_align, n*sizeof(Tp));
#if defined(__INTEL_COMPILER)
         ptr = (Tp*)malloc(bytes);
#else
         ptr = (Tp*)_mm_malloc(n*sizeof(Tp),alloc_align);
#endif
         if(!ptr) throw std::bad_alloc();
#endif

         return ptr;
       }

      void deallocate( Tp* p, std::size_t n) noexcept {
#ifdef __NVCOMPILER_CUDA__
         cudaFree((void *)p);
#elif !defined(DPCPP_BACKEND)

#if defined(__INTEL_COMPILER)
         free((void*)p);
#else
         _mm_free((void *)p);
#endif

#endif
       }
     };


using IntAllocator   = AlignedAllocator<int>;
using FloatAllocator = AlignedAllocator<float>;

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


constexpr int iparX     = 0;
constexpr int iparY     = 1;
constexpr int iparZ     = 2;
constexpr int iparIpt   = 3;
constexpr int iparPhi   = 4;
constexpr int iparTheta = 5;

template <typename T, typename Allocator, int n, int bSize>
struct MPNX {
   using DataType = T;

   static constexpr int N    = n;
   static constexpr int BS   = bSize;

   std::vector<T, Allocator> data;

   MPNX()                           : data(n*bSize){}
   MPNX(const size_t els)           : data(n*bSize*els){}
   MPNX(const std::vector<T, Allocator> data_) : data(data_){}
};

using MP1I    = MPNX<int,  IntAllocator,   1 , bsize>;
using MP3F    = MPNX<float,FloatAllocator, 3 , bsize>;
using MP6F    = MPNX<float,FloatAllocator, 6 , bsize>;
using MP3x3   = MPNX<float,FloatAllocator, 9 , bsize>;
using MP3x6   = MPNX<float,FloatAllocator, 18, bsize>;
using MP3x3SF = MPNX<float,FloatAllocator, 6 , bsize>;
using MP6x6SF = MPNX<float,FloatAllocator, 21, bsize>;
using MP6x6F  = MPNX<float,FloatAllocator, 36, bsize>;


template <typename MPNTp>
struct MPNXAccessor {
   typedef typename MPNTp::DataType T;

   static constexpr size_t bsz = MPNTp::BS;
   static constexpr size_t n   = MPNTp::N;
   static constexpr size_t stride = n*bsz;

   T* data_; //accessor field only for the data access, not allocated here

   MPNXAccessor() : data_(nullptr) {}
   MPNXAccessor(const MPNTp &v) : data_(const_cast<T*>(v.data.data())){
	}

   T* operator()(const size_t i = 0) const {return (data_ + stride*i);}
   T& operator()(const size_t i, const size_t j) const {return (data_ + stride*i)[j];}
   T& operator[](const size_t i) const {return data_[i];}

   // Restricted to MP3F (x,y,z) and MP6F (x,y,z,ipt,phi,theta) fields only: 
   template <int ipar, typename AccessedFieldTp = MPNTp>
   typename std::enable_if<(std::is_same<AccessedFieldTp, MP3F>::value and ipar < 3) or (std::is_same<AccessedFieldTp, MP6F>::value and ipar < 6), T>::type
   Get(size_t it, size_t id)  const { return (data_ + stride*id)[it + ipar*bsz]; }

   // Restricted to MP3F (x,y,z) fields only:   
   template <int ipar, typename AccessedFieldTp = MPNTp> 
   typename std::enable_if<std::is_same<AccessedFieldTp, MP3F>::value and ipar < 3, void>::type
   Set(size_t it, float val, size_t id)    { (data_ + stride*id)[it + ipar*bsz] = val; }

   // same as above but with a (shifted) raw pointer (and more generic)
   template <int ipar>
   static T Get(const T* local_data, size_t it)  { return local_data[it + ipar*bsz]; }  

   template <int ipar>
   static void Set(T* local_data, size_t it, T val)     { local_data[it + ipar*bsz] = val; }  

};

using MP6FAccessor   = MPNXAccessor<MP6F>;
using MP6x6SFAccessor= MPNXAccessor<MP6x6SF>;
using MP1IAccessor   = MPNXAccessor<MP1I>;

using MP3FAccessor   = MPNXAccessor<MP3F>;
using MP3x3SFAccessor= MPNXAccessor<MP3x3SF>;

using MP6x6FAccessor= MPNXAccessor<MP6x6F>;
using MP3x3Accessor = MPNXAccessor<MP3x3>;
using MP3x6Accessor = MPNXAccessor<MP3x6>;

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK() : par(), cov(), q() {}
  MPTRK(const size_t els) : par(els), cov(els), q(els) {}

  //  MP22I   hitidx;
};

struct MPTRKAccessor {
  MP6FAccessor    par;
  MP6x6SFAccessor cov;
  MP1IAccessor    q;
  MPTRKAccessor() : par(), cov(), q() {}
  MPTRKAccessor(const MPTRK &in) : par(in.par), cov(in.cov), q(in.q) {}
};


struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;

  MPHIT() : pos(), cov(){}
  MPHIT(const size_t els) : pos(els), cov(els) {}

};

struct MPHITAccessor {
  MP3FAccessor    pos;
  MP3x3SFAccessor cov;
  MPHITAccessor() : pos(), cov() {}
  MPHITAccessor(const MPHIT &in) : pos(in.pos), cov(in.cov) {}
};

//Pure host version: 

template <typename T, int N, int bSize>
struct MPNX_ {
   std::array<T,N*bSize> data;
};

using MP1I_    = MPNX_<int,   1 , bsize>;
using MP3F_    = MPNX_<float, 3 , bsize>;
using MP6F_    = MPNX_<float, 6 , bsize>;
using MP3x3SF_ = MPNX_<float, 6 , bsize>;
using MP6x6SF_ = MPNX_<float, 21, bsize>;

struct MPTRK_ {
  MP6F_    par;
  MP6x6SF_ cov;
  MP1I_    q;
  //  MP22I   hitidx;
};

struct MPHIT_ {
  MP3F_    pos;
  MP3x3SF_ cov;
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

MPTRK_* bTk(MPTRK_* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

inline const MPTRK_* bTkC(const MPTRK_* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

inline float q(const MP1I_* bq, size_t it){
  return (*bq).data[it];
}
//
inline float par(const MP6F_* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
inline float x    (const MP6F_* bpars, size_t it){ return par(bpars, it, 0); }
inline float y    (const MP6F_* bpars, size_t it){ return par(bpars, it, 1); }
inline float z    (const MP6F_* bpars, size_t it){ return par(bpars, it, 2); }
inline float ipt  (const MP6F_* bpars, size_t it){ return par(bpars, it, 3); }
inline float phi  (const MP6F_* bpars, size_t it){ return par(bpars, it, 4); }
inline float theta(const MP6F_* bpars, size_t it){ return par(bpars, it, 5); }
//
inline float par(const MPTRK_* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
inline float x    (const MPTRK_* btracks, size_t it){ return par(btracks, it, 0); }
inline float y    (const MPTRK_* btracks, size_t it){ return par(btracks, it, 1); }
inline float z    (const MPTRK_* btracks, size_t it){ return par(btracks, it, 2); }
inline float ipt  (const MPTRK_* btracks, size_t it){ return par(btracks, it, 3); }
inline float phi  (const MPTRK_* btracks, size_t it){ return par(btracks, it, 4); }
inline float theta(const MPTRK_* btracks, size_t it){ return par(btracks, it, 5); }
//
inline float par(const MPTRK_* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK_* btracks = bTkC(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
inline float x    (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
inline float y    (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
inline float z    (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
inline float ipt  (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
inline float phi  (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
inline float theta(const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }

inline const MPHIT_* bHit(const MPHIT_* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
inline const MPHIT_* bHit(const MPHIT_* hits, size_t ev, size_t ib, size_t lay) {
  return &(hits[lay + (ib + nb*ev)*nlayer]);
}
//
inline float pos(const MP3F_* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
inline float x(const MP3F_* hpos, size_t it)    { return pos(hpos, it, 0); }
inline float y(const MP3F_* hpos, size_t it)    { return pos(hpos, it, 1); }
inline float z(const MP3F_* hpos, size_t it)    { return pos(hpos, it, 2); }
//
inline float pos(const MPHIT_* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
inline float x(const MPHIT_* hits, size_t it)    { return pos(hits, it, 0); }
inline float y(const MPHIT_* hits, size_t it)    { return pos(hits, it, 1); }
inline float z(const MPHIT_* hits, size_t it)    { return pos(hits, it, 2); }
//
float pos(const MPHIT_* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT_* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
inline float x(const MPHIT_* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
inline float y(const MPHIT_* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
inline float z(const MPHIT_* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }

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

void convertTracks(MPTRK_* out,  const MPTRK* inp) {
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
    	  out[ib + nb*ie].par.data[it + ip*bsize] = inp->par.data[l*stride_par + ip*bsize + it];
    	}
    	//cov
    	for (size_t ip=0;ip<21;++ip) {
    	  out[ib + nb*ie].cov.data[it + ip*bsize] = inp->cov.data[l*stride_cov + ip*bsize + it];
    	}
    	//q
    	out[ib + nb*ie].q.data[it] = inp->q.data[l*stride_q + it];//fixme check
      }
    }
  }
  return;
}

MPHIT_* prepareHits(struct AHIT inputhit) {
  MPHIT_* result = new MPHIT_[nlayer*nevts*nb];

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

void convertHits(MPHIT_* out, const MPHIT* inp) {
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
        	  out[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = inp->pos.data[(lay+nlayer*l)*stride_pos + ip*bsize + it];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  out[lay+nlayer*(ib + nb*ie)].cov.data[it + ip*bsize] = inp->cov.data[(lay+nlayer*l)*stride_cov + ip*bsize +it];
        	}
        }
      }
    }
  }
  return;
}


#define N bsize 

template <size_t block_size = 1>
void MultHelixPropTranspEndcap(const MP6x6FAccessor &A, const MP6x6SFAccessor &B, MP6x6SFAccessor &C, const size_t lid, const size_t offset = 0) {
  const auto a = A(lid);
  const auto b = B(lid);
  auto c = C(lid);  
#pragma simd
  for (int n = offset; n < N; n += block_size)
  {
    auto temp00 = b[ 0*N+n] + a[ 2*N+n]*b[ 3*N+n] + a[ 3*N+n]*b[ 6*N+n] + a[ 4*N+n]*b[10*N+n] + a[ 5*N+n]*b[15*N+n];
    //auto temp01 = b[ 1*N+n] + a[ 2*N+n]*b[ 4*N+n] + a[ 3*N+n]*b[ 7*N+n] + a[ 4*N+n]*b[11*N+n] + a[ 5*N+n]*b[16*N+n];
    auto temp02 = b[ 3*N+n] + a[ 2*N+n]*b[ 5*N+n] + a[ 3*N+n]*b[ 8*N+n] + a[ 4*N+n]*b[12*N+n] + a[ 5*N+n]*b[17*N+n];
    auto temp03 = b[ 6*N+n] + a[ 2*N+n]*b[ 8*N+n] + a[ 3*N+n]*b[ 9*N+n] + a[ 4*N+n]*b[13*N+n] + a[ 5*N+n]*b[18*N+n];
    auto temp04 = b[10*N+n] + a[ 2*N+n]*b[12*N+n] + a[ 3*N+n]*b[13*N+n] + a[ 4*N+n]*b[14*N+n] + a[ 5*N+n]*b[19*N+n];
    auto temp05 = b[15*N+n] + a[ 2*N+n]*b[17*N+n] + a[ 3*N+n]*b[18*N+n] + a[ 4*N+n]*b[19*N+n] + a[ 5*N+n]*b[20*N+n];
    auto temp06 = b[ 1*N+n] + a[ 8*N+n]*b[ 3*N+n] + a[ 9*N+n]*b[ 6*N+n] + a[10*N+n]*b[10*N+n] + a[11*N+n]*b[15*N+n];
    auto temp07 = b[ 2*N+n] + a[ 8*N+n]*b[ 4*N+n] + a[ 9*N+n]*b[ 7*N+n] + a[10*N+n]*b[11*N+n] + a[11*N+n]*b[16*N+n];
    auto temp08 = b[ 4*N+n] + a[ 8*N+n]*b[ 5*N+n] + a[ 9*N+n]*b[ 8*N+n] + a[10*N+n]*b[12*N+n] + a[11*N+n]*b[17*N+n];
    auto temp09 = b[ 7*N+n] + a[ 8*N+n]*b[ 8*N+n] + a[ 9*N+n]*b[ 9*N+n] + a[10*N+n]*b[13*N+n] + a[11*N+n]*b[18*N+n];
    auto temp10 = b[11*N+n] + a[ 8*N+n]*b[12*N+n] + a[ 9*N+n]*b[13*N+n] + a[10*N+n]*b[14*N+n] + a[11*N+n]*b[19*N+n];
    auto temp11 = b[16*N+n] + a[ 8*N+n]*b[17*N+n] + a[ 9*N+n]*b[18*N+n] + a[10*N+n]*b[19*N+n] + a[11*N+n]*b[20*N+n];
    //auto temp12 = 0;
    //auto temp13 = 0;
    //auto temp14 = 0;
    //auto temp15 = 0;
    //auto temp16 = 0;
    //auto temp17 = 0;
    auto temp18 = b[ 6*N+n];
    auto temp19 = b[ 7*N+n];
    auto temp20 = b[ 8*N+n];
    auto temp21 = b[ 9*N+n];
    auto temp22 = b[13*N+n];
    auto temp23 = b[18*N+n];
    auto temp24 = a[26*N+n]*b[ 3*N+n] + a[27*N+n]*b[ 6*N+n] + b[10*N+n] + a[29*N+n]*b[15*N+n];
    auto temp25 = a[26*N+n]*b[ 4*N+n] + a[27*N+n]*b[ 7*N+n] + b[11*N+n] + a[29*N+n]*b[16*N+n];
    auto temp26 = a[26*N+n]*b[ 5*N+n] + a[27*N+n]*b[ 8*N+n] + b[12*N+n] + a[29*N+n]*b[17*N+n];
    auto temp27 = a[26*N+n]*b[ 8*N+n] + a[27*N+n]*b[ 9*N+n] + b[13*N+n] + a[29*N+n]*b[18*N+n];
    auto temp28 = a[26*N+n]*b[12*N+n] + a[27*N+n]*b[13*N+n] + b[14*N+n] + a[29*N+n]*b[19*N+n];
    auto temp29 = a[26*N+n]*b[17*N+n] + a[27*N+n]*b[18*N+n] + b[19*N+n] + a[29*N+n]*b[20*N+n];
    auto temp30 = b[15*N+n];
    auto temp31 = b[16*N+n];
    auto temp32 = b[17*N+n];
    auto temp33 = b[18*N+n];
    auto temp34 = b[19*N+n];
    auto temp35 = b[20*N+n];

    c[ 0*N+n] = temp00 + temp02*a[ 2*N+n] + temp03*a[ 3*N+n] + temp04*a[ 4*N+n] + temp05*a[ 5*N+n];
    c[ 1*N+n] = temp06 + temp08*a[ 2*N+n] + temp09*a[ 3*N+n] + temp10*a[ 4*N+n] + temp11*a[ 5*N+n];
    c[ 2*N+n] = temp07 + temp08*a[ 8*N+n] + temp09*a[ 9*N+n] + temp10*a[10*N+n] + temp11*a[11*N+n];
    c[ 3*N+n] = 0;
    c[ 4*N+n] = 0;
    c[ 5*N+n] = 0;
    c[ 6*N+n] = temp18 + temp20*a[ 2*N+n] + temp21*a[ 3*N+n] + temp22*a[ 4*N+n] + temp23*a[ 5*N+n];
    c[ 7*N+n] = temp19 + temp20*a[ 8*N+n] + temp21*a[ 9*N+n] + temp22*a[10*N+n] + temp23*a[11*N+n];
    c[ 8*N+n] = 0;
    c[ 9*N+n] = temp21;
    c[10*N+n] = temp24 + temp26*a[ 2*N+n] + temp27*a[ 3*N+n] + temp28*a[ 4*N+n] + temp29*a[ 5*N+n];
    c[11*N+n] = temp25 + temp26*a[ 8*N+n] + temp27*a[ 9*N+n] + temp28*a[10*N+n] + temp29*a[11*N+n];
    c[12*N+n] = 0;
    c[13*N+n] = temp27;
    c[14*N+n] = temp26*a[26*N+n] + temp27*a[27*N+n] + temp28 + temp29*a[29*N+n];
    c[15*N+n] = temp30 + temp32*a[ 2*N+n] + temp33*a[ 3*N+n] + temp34*a[ 4*N+n] + temp35*a[ 5*N+n];
    c[16*N+n] = temp31 + temp32*a[ 8*N+n] + temp33*a[ 9*N+n] + temp34*a[10*N+n] + temp35*a[11*N+n];
    c[17*N+n] = 0;
    c[18*N+n] = temp33;
    c[19*N+n] = temp32*a[26*N+n] + temp33*a[27*N+n] + temp34 + temp35*a[29*N+n];
    c[20*N+n] = temp35;
  }
}

template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type 
KalmanGainInv(const T* a, const T* b, std::array<T, 10> &c, const int n) {
  double det =
      ((a[0*N+n]+b[0*N+n])*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])))) -
      ((a[1*N+n]+b[1*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])))) +
      ((a[2*N+n]+b[2*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n]))));

  c[ 9] = 1.0 / det;

  c[ 0] =  c[ 9]*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])));
  c[ 1] =  -c[ 9]*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
  c[ 2] =  c[ 9]*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
  c[ 3] =  -c[ 9]*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])));
  c[ 4] =  c[ 9]*(((a[ 0*N+n]+b[ 0*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[2*N+n]+b[2*N+n])));
  c[ 5] =  -c[ 9]*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
  c[ 6] =  c[ 9]*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n])));
  c[ 7] =  -c[ 9]*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
  c[ 8] =  c[ 9]*(((a[ 0*N+n]+b[ 0*N+n]) *(a[6*N+n]+b[3*N+n])) - ((a[1*N+n]+b[1*N+n]) *(a[1*N+n]+b[1*N+n])));
}


template <size_t block_size = 1>
void KalmanUpdate(MP6x6SFAccessor  &trkErrAcc,
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
    const auto xin     = MP6FAccessor::Get<iparX>(inPar, it);
    const auto yin     = MP6FAccessor::Get<iparY>(inPar, it);
    const auto zin     = MP6FAccessor::Get<iparZ>(inPar, it);
    const auto ptin    = 1./MP6FAccessor::Get<iparIpt>(inPar, it);
    const auto phiin   = MP6FAccessor::Get<iparPhi>(inPar, it);
    const auto thetain = MP6FAccessor::Get<iparTheta>(inPar, it);
    const auto xout = MP3FAccessor::Get<iparX>(msP, it);
    const auto yout = MP3FAccessor::Get<iparY>(msP, it);
    const auto zout = MP3FAccessor::Get<iparZ>(msP, it);

    std::array<float, 10> temp{0.0f};

    KalmanGainInv(trkErr, hitErr, temp, it); 

    const auto kGain00 = trkErr[0*bsize+it]*temp[0] + trkErr[1*bsize+it]*temp[3] + trkErr[2*bsize+it]*temp[6];
    const auto kGain01 = trkErr[0*bsize+it]*temp[1] + trkErr[1*bsize+it]*temp[4] + trkErr[2*bsize+it]*temp[7];
    const auto kGain02 = trkErr[0*bsize+it]*temp[2] + trkErr[1*bsize+it]*temp[5] + trkErr[2*bsize+it]*temp[8];
    const auto kGain03 = trkErr[1*bsize+it]*temp[0] + trkErr[6*bsize+it]*temp[3] + trkErr[7*bsize+it]*temp[6];
    const auto kGain04 = trkErr[1*bsize+it]*temp[1] + trkErr[6*bsize+it]*temp[4] + trkErr[7*bsize+it]*temp[7];
    const auto kGain05 = trkErr[1*bsize+it]*temp[2] + trkErr[6*bsize+it]*temp[5] + trkErr[7*bsize+it]*temp[8];
    const auto kGain06 = trkErr[2*bsize+it]*temp[0] + trkErr[7*bsize+it]*temp[3] + trkErr[11*bsize+it]*temp[6];
    const auto kGain07 = trkErr[2*bsize+it]*temp[1] + trkErr[7*bsize+it]*temp[4] + trkErr[11*bsize+it]*temp[7];
    const auto kGain08 = trkErr[2*bsize+it]*temp[2] + trkErr[7*bsize+it]*temp[5] + trkErr[11*bsize+it]*temp[8];
    const auto kGain09 = trkErr[3*bsize+it]*temp[0] + trkErr[8*bsize+it]*temp[3] + trkErr[12*bsize+it]*temp[6];
    const auto kGain10 = trkErr[3*bsize+it]*temp[1] + trkErr[8*bsize+it]*temp[4] + trkErr[12*bsize+it]*temp[7];
    const auto kGain11 = trkErr[3*bsize+it]*temp[2] + trkErr[8*bsize+it]*temp[5] + trkErr[12*bsize+it]*temp[8];
    const auto kGain12 = trkErr[4*bsize+it]*temp[0] + trkErr[9*bsize+it]*temp[3] + trkErr[13*bsize+it]*temp[6];
    const auto kGain13 = trkErr[4*bsize+it]*temp[1] + trkErr[9*bsize+it]*temp[4] + trkErr[13*bsize+it]*temp[7];
    const auto kGain14 = trkErr[4*bsize+it]*temp[2] + trkErr[9*bsize+it]*temp[5] + trkErr[13*bsize+it]*temp[8];
    const auto kGain15 = trkErr[5*bsize+it]*temp[0] + trkErr[10*bsize+it]*temp[3] + trkErr[14*bsize+it]*temp[6];
    const auto kGain16 = trkErr[5*bsize+it]*temp[1] + trkErr[10*bsize+it]*temp[4] + trkErr[14*bsize+it]*temp[7];
    const auto kGain17 = trkErr[5*bsize+it]*temp[2] + trkErr[10*bsize+it]*temp[5] + trkErr[14*bsize+it]*temp[8];
  
    const auto xnew     = xin     + (kGain00*(xout-xin)) +(kGain01*(yout-yin));
    const auto ynew     = yin     + (kGain03*(xout-xin)) +(kGain04*(yout-yin));
    const auto znew     = zin     + (kGain06*(xout-xin)) +(kGain07*(yout-yin));
    const auto ptnew    = ptin    + (kGain09*(xout-xin)) +(kGain10*(yout-yin));
    const auto phinew   = phiin   + (kGain12*(xout-xin)) +(kGain13*(yout-yin));
    const auto thetanew = thetain + (kGain15*(xout-xin)) +(kGain16*(yout-yin));

    temp[0] = trkErr[0*bsize+it] - (kGain00*trkErr[0*bsize+it]+kGain01*trkErr[1*bsize+it]+kGain02*trkErr[2*bsize+it]);

    trkErr[0*bsize+it] = temp[0];

    temp[0] = trkErr[1*bsize+it] - (kGain00*trkErr[1*bsize+it]+kGain01*trkErr[6*bsize+it]+kGain02*trkErr[7*bsize+it]);
    temp[1] = trkErr[2*bsize+it] - (kGain00*trkErr[2*bsize+it]+kGain01*trkErr[7*bsize+it]+kGain02*trkErr[11*bsize+it]);
    temp[2] = trkErr[3*bsize+it] - (kGain00*trkErr[3*bsize+it]+kGain01*trkErr[8*bsize+it]+kGain02*trkErr[12*bsize+it]);
    temp[3] = trkErr[4*bsize+it] - (kGain00*trkErr[4*bsize+it]+kGain01*trkErr[9*bsize+it]+kGain02*trkErr[13*bsize+it]);
    temp[4] = trkErr[5*bsize+it] - (kGain00*trkErr[5*bsize+it]+kGain01*trkErr[10*bsize+it]+kGain02*trkErr[14*bsize+it]);
  
    temp[5] = trkErr[6*bsize+it] - (kGain03*trkErr[1*bsize+it]+kGain04*trkErr[6*bsize+it]+kGain05*trkErr[7*bsize+it]);

    trkErr[1*bsize+it] = temp[0];
    trkErr[6*bsize+it] = temp[5];

    temp[0] = trkErr[7*bsize+it] - (kGain03*trkErr[2*bsize+it]+kGain04*trkErr[7*bsize+it]+kGain05*trkErr[11*bsize+it]);
    temp[5] = trkErr[8*bsize+it] - (kGain03*trkErr[3*bsize+it]+kGain04*trkErr[8*bsize+it]+kGain05*trkErr[12*bsize+it]);
    temp[6] = trkErr[9*bsize+it] - (kGain03*trkErr[4*bsize+it]+kGain04*trkErr[9*bsize+it]+kGain05*trkErr[13*bsize+it]);
    temp[7] = trkErr[10*bsize+it] - (kGain03*trkErr[5*bsize+it]+kGain04*trkErr[10*bsize+it]+kGain05*trkErr[14*bsize+it]);
  
    temp[8] = trkErr[11*bsize+it] - (kGain06*trkErr[2*bsize+it]+kGain07*trkErr[7*bsize+it]+kGain08*trkErr[11*bsize+it]);

    trkErr[2*bsize+it]  = temp[1];
    trkErr[7*bsize+it]  = temp[0];
    trkErr[11*bsize+it] = temp[8];

    temp[1] = trkErr[12*bsize+it] - (kGain06*trkErr[3*bsize+it]+kGain07*trkErr[8*bsize+it]+kGain08*trkErr[12*bsize+it]);
    temp[0] = trkErr[13*bsize+it] - (kGain06*trkErr[4*bsize+it]+kGain07*trkErr[9*bsize+it]+kGain08*trkErr[13*bsize+it]);
    temp[8] = trkErr[14*bsize+it] - (kGain06*trkErr[5*bsize+it]+kGain07*trkErr[10*bsize+it]+kGain08*trkErr[14*bsize+it]);
    temp[9] = trkErr[15*bsize+it] - (kGain09*trkErr[3*bsize+it]+kGain10*trkErr[8*bsize+it]+kGain11*trkErr[12*bsize+it]);

    trkErr[3*bsize+it]  = temp[2];
    trkErr[8*bsize+it]  = temp[5];
    trkErr[12*bsize+it] = temp[1];
    trkErr[15*bsize+it] = temp[9];

    temp[2] = trkErr[16*bsize+it] - (kGain09*trkErr[4*bsize+it]+kGain10*trkErr[9*bsize+it]+kGain11*trkErr[13*bsize+it]);

    trkErr[16*bsize+it] = temp[2];

    temp[5] = trkErr[17*bsize+it] - (kGain09*trkErr[5*bsize+it]+kGain10*trkErr[10*bsize+it]+kGain11*trkErr[14*bsize+it]);

    trkErr[17*bsize+it] = temp[5];
  
    temp[1] = trkErr[18*bsize+it] - (kGain12*trkErr[4*bsize+it]+kGain13*trkErr[9*bsize+it]+kGain14*trkErr[13*bsize+it]);

    trkErr[4*bsize+it]  = temp[3];
    trkErr[9*bsize+it]  = temp[6];
    trkErr[13*bsize+it] = temp[0];
    trkErr[18*bsize+it] = temp[1];

    temp[9] = trkErr[19*bsize+it] - (kGain12*trkErr[5*bsize+it]+kGain13*trkErr[10*bsize+it]+kGain14*trkErr[14*bsize+it]);

    trkErr[19*bsize+it] = temp[9];

    temp[9] = trkErr[20*bsize+it] - (kGain15*trkErr[5*bsize+it]+kGain16*trkErr[10*bsize+it]+kGain17*trkErr[14*bsize+it]);

    trkErr[10*bsize+it] = temp[7];
    trkErr[5*bsize+it]  = temp[4];
    trkErr[14*bsize+it] = temp[8];
    trkErr[20*bsize+it] = temp[9];

    MP6FAccessor::Set<iparX>(inPar,it, xnew);
    MP6FAccessor::Set<iparY>(inPar,it, ynew);
    MP6FAccessor::Set<iparZ>(inPar,it, znew);
    MP6FAccessor::Set<iparIpt>(inPar,it, ptnew);
    MP6FAccessor::Set<iparPhi>(inPar,it, phinew);
    MP6FAccessor::Set<iparTheta>(inPar,it, thetanew);
  }

  return;
}


const float kfact = 100/3.8;

template <size_t block_size = 1>
void propagateToZ(const MPTRKAccessor &btracks,
		  const MPHITAccessor &bhits,
                  MPTRKAccessor  &obtracks,
                  MP6x6FAccessor &errorPropAcc,  
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

  MultHelixPropTranspEndcap<block_size>(errorPropAcc, btracks.cov, obtracks.cov, lid, offset);

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
   
   auto trkNPtr = prepareTracksN(inputtrk);
   std::unique_ptr<MPTRKAccessor> trkNaccPtr(new MPTRKAccessor(*trkNPtr));

   auto hitNPtr = prepareHitsN(inputhit);
   std::unique_ptr<MPHITAccessor> hitNaccPtr(new MPHITAccessor(*hitNPtr));

   std::unique_ptr<MPTRK> outtrkNPtr(new MPTRK(nevts*nb));
   std::unique_ptr<MPTRKAccessor> outtrkNaccPtr(new MPTRKAccessor(*outtrkNPtr));

   std::unique_ptr<MP6x6F>  errPropPtr(new MP6x6F(nevts*nb));
   std::unique_ptr<MP6x6FAccessor>  errorPropAccPtr(new MP6x6FAccessor(*errPropPtr));

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");
   
   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   auto wall_start = std::chrono::high_resolution_clock::now();
#ifdef __NVCOMPILER_CUDA__
   constexpr size_t blk_sz = bsize;
#else
   constexpr size_t blk_sz = 1;
#endif   
   auto policy = std::execution::par_unseq;

   for(itr=0; itr<NITER; itr++) {

     const int outer_loop_range = nevts*nb*blk_sz;//z,y,x
     const int nbxblk_sz        = nb*blk_sz;//y,x

     std::for_each(policy,
                   counting_iterator(0),
                   counting_iterator(outer_loop_range),
                   [=,  &trkNacc = *trkNaccPtr, 
			&hitNacc = *hitNaccPtr, 
			&outtrkNacc = *outtrkNaccPtr,
			&errorPropAcc = *errorPropAccPtr] (auto ii) {
                   const size_t ie                = ii / nbxblk_sz;//z
                   const size_t ibt               = ii - ie*nbxblk_sz;//
                   const size_t ib                = ibt / blk_sz;//y  
                   const size_t inner_loop_offset = ibt - ib*blk_sz;//x
                   const size_t li  = ib+nb*ie;
                   for(size_t layer=0; layer<nlayer; ++layer) {
                     const size_t lli = layer+li*nlayer;
                     //
                     propagateToZ<blk_sz>(trkNacc, hitNacc, outtrkNacc, errorPropAcc, li, lli, inner_loop_offset);
                     KalmanUpdate<blk_sz>(outtrkNacc.cov, outtrkNacc.par, hitNacc, li, lli, inner_loop_offset);
                   }

                   });
   } //end of itr loop

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   MPHIT_* hit    = prepareHits(inputhit);
   MPTRK_* outtrk = (MPTRK_*) malloc(nevts*nb*sizeof(MPTRK_));

   convertTracks(outtrk, outtrkNPtr.get());
   convertHits(hit, hitNPtr.get());

   float avgx = 0, avgy = 0, avgz = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float pt_    = 1./ipt(outtrk,ie,it);
       float phi_   = phi(outtrk,ie,it);
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

   delete [] hit;
   delete [] outtrk;


   return 0;
}
