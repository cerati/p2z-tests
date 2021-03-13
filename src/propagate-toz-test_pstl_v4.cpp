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
#if defined(__NVCOMPILER_CUDA__)
#define bsize 1
#else
#define bsize 128
#endif//__NVCOMPILER_CUDA__
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

enum class FieldOrder{P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER,
                      P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER,
                      P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER};

using IntAllocator   = AlignedAllocator<int>;
using FloatAllocator = AlignedAllocator<float>;

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

   const int nTrks;//note that bSize is a tuning parameter!
   const int nEvts;
   const int nLayers;

   std::vector<T, Allocator> data;

   MPNX() : nTrks(bSize), nEvts(0), nLayers(0), data(n*bSize){}

   MPNX(const int ntrks_, const int nevts_, const int nlayers_ = 1) :
      nTrks(ntrks_),
      nEvts(nevts_),
      nLayers(nlayers_),
      data(n*nTrks*nEvts*nLayers){
   }

   MPNX(const std::vector<T, Allocator> data_, const int ntrks_, const int nevts_, const int nlayers_ = 1) :
      nTrks(ntrks_),
      nEvts(nevts_),
      nLayers(nlayers_),
      data(data_) {
     if(data_.size() > n*nTrks*nEvts*nLayers) {std::cerr << "Incorrect dim parameters."; }
   }
};

using MP1I    = MPNX<int,  IntAllocator,   1 , bsize>;
using MP3F    = MPNX<float,FloatAllocator, 3 , bsize>;
using MP6F    = MPNX<float,FloatAllocator, 6 , bsize>;
using MP3x3   = MPNX<float,FloatAllocator, 9 , bsize>;
using MP3x6   = MPNX<float,FloatAllocator, 18, bsize>;
using MP3x3SF = MPNX<float,FloatAllocator, 6 , bsize>;
using MP6x6SF = MPNX<float,FloatAllocator, 21, bsize>;
using MP6x6F  = MPNX<float,FloatAllocator, 36, bsize>;

template <typename MPNTp, FieldOrder Order = FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER>
struct MPNXAccessor {
   typedef typename MPNTp::DataType T;

   static constexpr int bsz = MPNTp::BS;
   static constexpr int n   = MPNTp::N;

   const int nTrkB;
   const int nEvts;
   const int nLayers;

   const int NevtsNtbBsz;

   const int stride;

   T* data_; //accessor field only for the data access, not allocated here

   MPNXAccessor() : nTrkB(0), nEvts(0), nLayers(0), NevtsNtbBsz(0), stride(0), data_(nullptr){}
   MPNXAccessor(const MPNTp &v) :
        nTrkB(v.nTrks / bsz),
        nEvts(v.nEvts),
        nLayers(v.nLayers),
        NevtsNtbBsz(nEvts*nTrkB*bsz),
        stride(Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER ? bsz*nTrkB*nEvts*nLayers  :
              (Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER ? bsz*nTrkB*nEvts*n : n*bsz*nLayers)),
        data_(const_cast<T*>(v.data.data())){
	 }

   T& operator[](const int idx) const {return data_[idx];}

   T& operator()(const int lid, const int tid, const int b = 0, const int layer = 0) const {
     if      constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return data_[lid*stride + layer*NevtsNtbBsz + tid*bsz + b];//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return data_[layer*stride + lid*NevtsNtbBsz + tid*bsz + b];
     else if constexpr (Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return data_[tid*stride+layer*n*bsz+lid*bsz+b];
   }//i is the internal dof index

};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK() : par(), cov(), q() {}
  MPTRK(const int ntrks_, const int nevts_) : par(ntrks_, nevts_), cov(ntrks_, nevts_), q(ntrks_, nevts_) {}

  //  MP22I   hitidx;
};

template <FieldOrder Order>
struct MPTRKAccessor {
  using MP6FAccessor   = MPNXAccessor<MP6F,    Order>;
  using MP6x6SFAccessor= MPNXAccessor<MP6x6SF, Order>;
  using MP1IAccessor   = MPNXAccessor<MP1I,    Order>;

  MP6FAccessor    par;
  MP6x6SFAccessor cov;
  MP1IAccessor    q;

  MPTRKAccessor() : par(), cov(), q() {}
  MPTRKAccessor(const MPTRK &in) : par(in.par), cov(in.cov), q(in.q) {}
};

template<FieldOrder order>
std::shared_ptr<MPTRK> prepareTracksN(struct ATRK inputtrk) {

  auto result = std::make_shared<MPTRK>(ntrks, nevts);
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> rA(new MPTRKAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	  //par
    	  for (size_t ip=0;ip<6;++ip) {
          rA->par(ip, tid, it) = (1+smear*randn(0,1))*inputtrk.par[ip];
    	  }
    	  //cov
    	  for (size_t ip=0;ip<21;++ip) {
          rA->cov(ip, tid, it) = (1+smear*randn(0,1))*inputtrk.cov[ip];
    	  }
    	  //q
        rA->q(0, tid, it) = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);
      }
    }
  }
  return std::move(result);
}

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;

  MPHIT() : pos(), cov(){}
  MPHIT(const int ntrks_, const int nevts_, const int nlayers_) : pos(ntrks_, nevts_, nlayers_), cov(ntrks_, nevts_, nlayers_) {}

};

template <FieldOrder Order>
struct MPHITAccessor {
  using MP3FAccessor   = MPNXAccessor<MP3F,    Order>;
  using MP3x3SFAccessor= MPNXAccessor<MP3x3SF, Order>;

  MP3FAccessor    pos;
  MP3x3SFAccessor cov;

  MPHITAccessor() : pos(), cov() {}
  MPHITAccessor(const MPHIT &in) : pos(in.pos), cov(in.cov) {}
};

template<FieldOrder order>
std::shared_ptr<MPHIT> prepareHitsN(struct AHIT inputhit) {
  auto result = std::make_shared<MPHIT>(ntrks, nevts, nlayer);
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> rA(new MPHITAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
            rA->pos(ip, tid, it, lay) = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
            rA->cov(ip, tid, it, lay) = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return std::move(result);
}



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

template<FieldOrder order>
void convertTracks(MPTRK_* out,  const MPTRK* inp) {
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> inpA(new MPTRKAccessor<order>(*inp));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	  //par
    	  for (size_t ip=0;ip<6;++ip) {
    	    out[tid].par.data[it + ip*bsize] = inpA->par(ip, tid, it);
    	  }
    	  //cov
    	  for (size_t ip=0;ip<21;++ip) {
    	    out[tid].cov.data[it + ip*bsize] = inpA->cov(ip, tid, it);
    	  }
    	  //q
    	  out[tid].q.data[it] = inpA->q(0, tid, it);//fixme check
      }
    }
  }
  return;
}

template<FieldOrder order>
void convertHits(MPHIT_* out, const MPHIT* inp) {
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> inpA(new MPHITAccessor<order>(*inp));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
            out[lay+nlayer*tid].pos.data[it + ip*bsize] = inpA->pos(ip, tid, it, lay);
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
            out[lay+nlayer*tid].cov.data[it + ip*bsize] = inpA->cov(ip, tid, it, lay);
        	}
        }
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

template<typename T, typename Accessor1, typename Accessor2>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
KalmanGainInv(const Accessor1 &a, const Accessor2 &b, std::array<T, 10> &c, const int tid, const int it, const int lay) {
  double det =
      ((a(0, tid, it)+b(0, tid, it, lay))*(((a(6, tid, it)+b(3, tid, it, lay)) *(a(11,tid, it)+b(5, tid, it, lay))) - ((a(7, tid, it)+b(4, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))))) -
      ((a(1, tid, it)+b(1, tid, it, lay))*(((a(1, tid, it)+b(1, tid, it, lay)) *(a(11,tid, it)+b(5, tid, it, lay))) - ((a(7, tid, it)+b(4, tid, it, lay)) *(a(2, tid, it)+b(2, tid, it, lay))))) +
      ((a(2, tid, it)+b(2, tid, it, lay))*(((a(1, tid, it)+b(1, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(6, tid, it)+b(3, tid, it, lay)))));

  c[ 9] = 1.0 / det;

  c[ 0] =   c[ 9]*(((a(6, tid, it)+b(3, tid, it, lay)) *(a(11,tid, it)+b(5, tid, it, lay))) - ((a(7, tid, it)+b(4, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))));
  c[ 1] =  -c[ 9]*(((a(1, tid, it)+b(1, tid, it, lay)) *(a(11,tid, it)+b(5, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))));
  c[ 2] =   c[ 9]*(((a(1, tid, it)+b(1, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))));
  c[ 3] =  -c[ 9]*(((a(1, tid, it)+b(1, tid, it, lay)) *(a(11,tid, it)+b(5, tid, it, lay))) - ((a(7, tid, it)+b(4, tid, it, lay)) *(a(2, tid, it)+b(2, tid, it, lay))));
  c[ 4] =   c[ 9]*(((a(0, tid, it)+b(0, tid, it, lay)) *(a(11,tid, it)+b(5, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(2, tid, it)+b(2, tid, it, lay))));
  c[ 5] =  -c[ 9]*(((a(0, tid, it)+b(0, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(1, tid, it)+b(1, tid, it, lay))));
  c[ 6] =   c[ 9]*(((a(1, tid, it)+b(1, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(6, tid, it)+b(3, tid, it, lay))));
  c[ 7] =  -c[ 9]*(((a(0, tid, it)+b(0, tid, it, lay)) *(a(7, tid, it)+b(4, tid, it, lay))) - ((a(2, tid, it)+b(2, tid, it, lay)) *(a(1, tid, it)+b(1, tid, it, lay))));
  c[ 8] =   c[ 9]*(((a(0, tid, it)+b(0, tid, it, lay)) *(a(6, tid, it)+b(3, tid, it, lay))) - ((a(1, tid, it)+b(1, tid, it, lay)) *(a(1, tid, it)+b(1, tid, it, lay))));
}

template <class MPTRKAccessors, class MPHITAccessors, size_t block_size = 1>
void KalmanUpdate(MPTRKAccessors       &obtracks,
		              const MPHITAccessors &bhits,
                  const int tid,
                  const int lay) {
  using MP6FAccessor_    = typename MPTRKAccessors::MP6FAccessor;
  using MP6x6SFAccessor_ = typename MPTRKAccessors::MP6x6SFAccessor;
  using MP3x3SFAccessor_ = typename MPHITAccessors::MP3x3SFAccessor;
  using MP3FAccessor_    = typename MPHITAccessors::MP3FAccessor;

  const MP3x3SFAccessor_ &hitErr   = bhits.cov;
  const MP3FAccessor_    &msP      = bhits.pos;

  MP6x6SFAccessor_  &trkErr = obtracks.cov;
  MP6FAccessor_     &inPar  = obtracks.par;

#pragma simd
  for (int it = 0; it < block_size; it++) {
    const auto xin     = inPar(iparX, tid, it);
    const auto yin     = inPar(iparY, tid, it);
    const auto zin     = inPar(iparZ, tid, it);
    const auto ptin    = 1. / inPar(iparIpt, tid, it);
    const auto phiin   = inPar(iparPhi, tid, it);
    const auto thetain = inPar(iparTheta, tid, it);
    const auto xout    = msP(iparX, tid, it, lay);
    const auto yout    = msP(iparY, tid, it, lay);

    std::array<float, 10> temp{0.0f};

    KalmanGainInv(trkErr, hitErr, temp, tid, it, lay);

    const auto kGain00 = trkErr(0, tid, it)*temp[0] + trkErr(1, tid, it)*temp[3] + trkErr(2, tid, it)*temp[6];
    const auto kGain01 = trkErr(0, tid, it)*temp[1] + trkErr(1, tid, it)*temp[4] + trkErr(2, tid, it)*temp[7];
    const auto kGain02 = trkErr(0, tid, it)*temp[2] + trkErr(1, tid, it)*temp[5] + trkErr(2, tid, it)*temp[8];
    const auto kGain03 = trkErr(1, tid, it)*temp[0] + trkErr(6, tid, it)*temp[3] + trkErr(7, tid, it)*temp[6];
    const auto kGain04 = trkErr(1, tid, it)*temp[1] + trkErr(6, tid, it)*temp[4] + trkErr(7, tid, it)*temp[7];
    const auto kGain05 = trkErr(1, tid, it)*temp[2] + trkErr(6, tid, it)*temp[5] + trkErr(7, tid, it)*temp[8];
    const auto kGain06 = trkErr(2, tid, it)*temp[0] + trkErr(7, tid, it)*temp[3] + trkErr(11, tid, it)*temp[6];
    const auto kGain07 = trkErr(2, tid, it)*temp[1] + trkErr(7, tid, it)*temp[4] + trkErr(11, tid, it)*temp[7];
    const auto kGain08 = trkErr(2, tid, it)*temp[2] + trkErr(7, tid, it)*temp[5] + trkErr(11, tid, it)*temp[8];
    const auto kGain09 = trkErr(3, tid, it)*temp[0] + trkErr(8, tid, it)*temp[3] + trkErr(12, tid, it)*temp[6];
    const auto kGain10 = trkErr(3, tid, it)*temp[1] + trkErr(8, tid, it)*temp[4] + trkErr(12, tid, it)*temp[7];
    const auto kGain11 = trkErr(3, tid, it)*temp[2] + trkErr(8, tid, it)*temp[5] + trkErr(12, tid, it)*temp[8];
    const auto kGain12 = trkErr(4, tid, it)*temp[0] + trkErr(9, tid, it)*temp[3] + trkErr(13, tid, it)*temp[6];
    const auto kGain13 = trkErr(4, tid, it)*temp[1] + trkErr(9, tid, it)*temp[4] + trkErr(13, tid, it)*temp[7];
    const auto kGain14 = trkErr(4, tid, it)*temp[2] + trkErr(9, tid, it)*temp[5] + trkErr(13, tid, it)*temp[8];
    const auto kGain15 = trkErr(5, tid, it)*temp[0] + trkErr(10, tid, it)*temp[3] + trkErr(14, tid, it)*temp[6];
    const auto kGain16 = trkErr(5, tid, it)*temp[1] + trkErr(10, tid, it)*temp[4] + trkErr(14, tid, it)*temp[7];
    const auto kGain17 = trkErr(5, tid, it)*temp[2] + trkErr(10, tid, it)*temp[5] + trkErr(14, tid, it)*temp[8];

    const auto xnew     = xin     + (kGain00*(xout-xin)) +(kGain01*(yout-yin));
    const auto ynew     = yin     + (kGain03*(xout-xin)) +(kGain04*(yout-yin));
    const auto znew     = zin     + (kGain06*(xout-xin)) +(kGain07*(yout-yin));
    const auto ptnew    = ptin    + (kGain09*(xout-xin)) +(kGain10*(yout-yin));
    const auto phinew   = phiin   + (kGain12*(xout-xin)) +(kGain13*(yout-yin));
    const auto thetanew = thetain + (kGain15*(xout-xin)) +(kGain16*(yout-yin));

    temp[0] = trkErr(0, tid, it) - (kGain00*trkErr(0, tid, it)+kGain01*trkErr(1, tid, it)+kGain02*trkErr(2, tid, it));

    trkErr(0, tid, it) = temp[0];

    temp[0] = trkErr(1, tid, it) - (kGain00*trkErr(1, tid, it)+kGain01*trkErr(6, tid, it)+kGain02*trkErr(7, tid, it));
    temp[1] = trkErr(2, tid, it) - (kGain00*trkErr(2, tid, it)+kGain01*trkErr(7, tid, it)+kGain02*trkErr(11, tid, it));
    temp[2] = trkErr(3, tid, it) - (kGain00*trkErr(3, tid, it)+kGain01*trkErr(8, tid, it)+kGain02*trkErr(12, tid, it));
    temp[3] = trkErr(4, tid, it) - (kGain00*trkErr(4, tid, it)+kGain01*trkErr(9, tid, it)+kGain02*trkErr(13, tid, it));
    temp[4] = trkErr(5, tid, it) - (kGain00*trkErr(5, tid, it)+kGain01*trkErr(10, tid, it)+kGain02*trkErr(14, tid, it));

    temp[5] = trkErr(6, tid, it) - (kGain03*trkErr(1, tid, it)+kGain04*trkErr(6, tid, it)+kGain05*trkErr(7, tid, it));

    trkErr(1, tid, it) = temp[0];
    trkErr(6, tid, it) = temp[5];

    temp[0] = trkErr(7, tid, it) - (kGain03*trkErr(2, tid, it)+kGain04*trkErr(7, tid, it)+kGain05*trkErr(11, tid, it));
    temp[5] = trkErr(8, tid, it) - (kGain03*trkErr(3, tid, it)+kGain04*trkErr(8, tid, it)+kGain05*trkErr(12, tid, it));
    temp[6] = trkErr(9, tid, it) - (kGain03*trkErr(4, tid, it)+kGain04*trkErr(9, tid, it)+kGain05*trkErr(13, tid, it));
    temp[7] = trkErr(10, tid, it) - (kGain03*trkErr(5, tid, it)+kGain04*trkErr(10, tid, it)+kGain05*trkErr(14, tid, it));

    temp[8] = trkErr(11, tid, it) - (kGain06*trkErr(2, tid, it)+kGain07*trkErr(7, tid, it)+kGain08*trkErr(11, tid, it));

    trkErr(2, tid, it)  = temp[1];
    trkErr(7, tid, it)  = temp[0];
    trkErr(11, tid, it) = temp[8];

    temp[1] = trkErr(12, tid, it) - (kGain06*trkErr(3, tid, it)+kGain07*trkErr(8, tid, it)+kGain08*trkErr(12, tid, it));
    temp[0] = trkErr(13, tid, it) - (kGain06*trkErr(4, tid, it)+kGain07*trkErr(9, tid, it)+kGain08*trkErr(13, tid, it));
    temp[8] = trkErr(14, tid, it) - (kGain06*trkErr(5, tid, it)+kGain07*trkErr(10, tid, it)+kGain08*trkErr(14, tid, it));
    temp[9] = trkErr(15, tid, it) - (kGain09*trkErr(3, tid, it)+kGain10*trkErr(8, tid, it)+kGain11*trkErr(12, tid, it));

    trkErr(3, tid, it)  = temp[2];
    trkErr(8, tid, it)  = temp[5];
    trkErr(12, tid, it) = temp[1];
    trkErr(15, tid, it) = temp[9];

    temp[2] = trkErr(16, tid, it) - (kGain09*trkErr(4, tid, it)+kGain10*trkErr(9, tid, it)+kGain11*trkErr(13, tid, it));

    trkErr(16, tid, it) = temp[2];

    temp[5] = trkErr(17, tid, it) - (kGain09*trkErr(5, tid, it)+kGain10*trkErr(10, tid, it)+kGain11*trkErr(14, tid, it));

    trkErr(17, tid, it) = temp[5];

    temp[1] = trkErr(18, tid, it) - (kGain12*trkErr(4, tid, it)+kGain13*trkErr(9, tid, it)+kGain14*trkErr(13, tid, it));

    trkErr(4, tid, it)  = temp[3];
    trkErr(9, tid, it)  = temp[6];
    trkErr(13, tid, it) = temp[0];
    trkErr(18, tid, it) = temp[1];

    temp[9] = trkErr(19, tid, it) - (kGain12*trkErr(5, tid, it)+kGain13*trkErr(10, tid, it)+kGain14*trkErr(14, tid, it));

    trkErr(19, tid, it) = temp[9];

    temp[9] = trkErr(20, tid, it) - (kGain15*trkErr(5, tid, it)+kGain16*trkErr(10, tid, it)+kGain17*trkErr(14, tid, it));

    trkErr(10, tid, it) = temp[7];
    trkErr(5, tid, it)  = temp[4];
    trkErr(14, tid, it) = temp[8];
    trkErr(20, tid, it) = temp[9];

    inPar(iparX,tid, it)     = xnew;
    inPar(iparY,tid, it)     = ynew;
    inPar(iparZ,tid, it)     = znew;
    inPar(iparIpt,tid, it)   = ptnew;
    inPar(iparPhi,tid, it)   = phinew;
    inPar(iparTheta,tid, it) = thetanew;
  }

  return;
}


const float kfact = 100/3.8;

template <class MPTRKAccessors, class MPHITAccessors, size_t block_size = 1>
void propagateToZ(MPTRKAccessors       &obtracks,
                  const MPTRKAccessors &btracks,
                  const MPHITAccessors &bhits,
                  const int tid,
                  const int lay) {

  using MP6FAccessor_    = typename MPTRKAccessors::MP6FAccessor;
  using MP1IAccessor_    = typename MPTRKAccessors::MP1IAccessor;
  using MP6x6SFAccessor_ = typename MPTRKAccessors::MP6x6SFAccessor;
  using MP3FAccessor_    = typename MPHITAccessors::MP3FAccessor;

  const MP6FAccessor_ &inPar    = btracks.par;
  const MP1IAccessor_ &inChg    = btracks.q  ;
  const MP6x6SFAccessor_ &inErr = btracks.cov;

  const MP3FAccessor_ &msP      = bhits.pos;

  MP6x6SFAccessor_ &outErr    = obtracks.cov;
  MP6FAccessor_    &outPar    = obtracks.par;


#pragma simd
  for (int it = 0;it < block_size; it++) {
    const float zout = msP(iparZ, tid, it, lay);
    const float k    = inChg(0, tid, it)*kfact;//100/3.8;
    const float deltaZ = zout - inPar(iparZ, tid, it);
    const float pt     = inPar(iparIpt, tid, it);
    const float cosP   = cosf(inPar(iparPhi, tid, it));//inPar(iparPhi, tid, it)
    const float sinP   = sinf(inPar(iparPhi, tid, it));
    const float cosT   = cosf(inPar(iparTheta, tid, it));
    const float sinT   = sinf(inPar(iparTheta, tid, it));

    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*pt*icosTk;

    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;

    outPar(iparX,tid, it) = inPar(iparX, tid, it) + k*(pxin*sina - pyin*(1.-cosa));
    outPar(iparY,tid, it) = inPar(iparY, tid, it) + k*(pyin*sina + pxin*(1.-cosa));
    outPar(iparZ,tid, it) = zout;

    outPar(iparIpt,tid, it)   = inPar(iparIpt, tid, it);
    outPar(iparPhi,tid, it)   = inPar(iparPhi, tid, it) + alpha;
    outPar(iparTheta,tid, it) = inPar(iparTheta, tid, it);

    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);

    auto errorProp2  = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    auto errorProp3  = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.-cCosPsina))*(pt*pt);
    auto errorProp4  = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    auto errorProp5  = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*icosT);
    auto errorProp8  = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    auto errorProp9  = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.-cCosPsina))*(pt*pt);
    auto errorProp10 = (k*pt)*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    auto errorProp11 = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    auto errorProp26 = -inPar(iparIpt, tid, it)*sinT*(icosTk);
    auto errorProp27 = sinT*deltaZ*(icosTk);
    auto errorProp29 = inPar(iparIpt, tid, it)*deltaZ*(icosT*icosTk);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto temp00 = inErr(0, tid, it) + errorProp2 *inErr(3, tid, it) + errorProp3 *inErr(6, tid, it) + errorProp4  *inErr(10,tid, it) + errorProp5  *inErr(15,tid, it);
    auto temp02 = inErr(3, tid, it) + errorProp2 *inErr(5, tid, it) + errorProp3 *inErr(8, tid, it) + errorProp4  *inErr(12,tid, it) + errorProp5  *inErr(17,tid, it);
    auto temp03 = inErr(6, tid, it) + errorProp2 *inErr(8, tid, it) + errorProp3 *inErr(9, tid, it) + errorProp4  *inErr(13,tid, it) + errorProp5  *inErr(18,tid, it);
    auto temp04 = inErr(10,tid, it) + errorProp2 *inErr(12,tid, it) + errorProp3 *inErr(13,tid, it) + errorProp4  *inErr(14,tid, it) + errorProp5  *inErr(19,tid, it);
    auto temp05 = inErr(15,tid, it) + errorProp2 *inErr(17,tid, it) + errorProp3 *inErr(18,tid, it) + errorProp4  *inErr(19,tid, it) + errorProp5  *inErr(20,tid, it);
    auto temp06 = inErr(1, tid, it) + errorProp8 *inErr(3, tid, it) + errorProp9 *inErr(6, tid, it) + errorProp10 *inErr(10,tid, it) + errorProp11 *inErr(15,tid, it);
    auto temp07 = inErr(2, tid, it) + errorProp8 *inErr(4, tid, it) + errorProp9 *inErr(7, tid, it) + errorProp10 *inErr(11,tid, it) + errorProp11 *inErr(16,tid, it);
    auto temp08 = inErr(4, tid, it) + errorProp8 *inErr(5, tid, it) + errorProp9 *inErr(8, tid, it) + errorProp10 *inErr(12,tid, it) + errorProp11 *inErr(17,tid, it);
    auto temp09 = inErr(7, tid, it) + errorProp8 *inErr(8, tid, it) + errorProp9 *inErr(9, tid, it) + errorProp10 *inErr(13,tid, it) + errorProp11 *inErr(18,tid, it);
    auto temp10 = inErr(11,tid, it) + errorProp8 *inErr(12,tid, it) + errorProp9 *inErr(13,tid, it) + errorProp10 *inErr(14,tid, it) + errorProp11 *inErr(19,tid, it);
    auto temp11 = inErr(16,tid, it) + errorProp8 *inErr(17,tid, it) + errorProp9 *inErr(18,tid, it) + errorProp10 *inErr(19,tid, it) + errorProp11 *inErr(20,tid, it);

    auto temp24 = errorProp26 *inErr(3, tid, it) + errorProp27 *inErr(6, tid, it) + inErr(10, tid, it) + errorProp29 *inErr(15, tid, it);
    auto temp25 = errorProp26 *inErr(4, tid, it) + errorProp27 *inErr(7, tid, it) + inErr(11, tid, it) + errorProp29 *inErr(16, tid, it);
    auto temp26 = errorProp26 *inErr(5, tid, it) + errorProp27 *inErr(8, tid, it) + inErr(12, tid, it) + errorProp29 *inErr(17, tid, it);
    auto temp27 = errorProp26 *inErr(8, tid, it) + errorProp27 *inErr(9, tid, it) + inErr(13, tid, it) + errorProp29 *inErr(18, tid, it);
    auto temp28 = errorProp26 *inErr(12, tid, it) + errorProp27 *inErr(13, tid, it) + inErr(14, tid, it) + errorProp29 *inErr(19, tid, it);
    auto temp29 = errorProp26 *inErr(17, tid, it) + errorProp27 *inErr(18, tid, it) + inErr(19, tid, it) + errorProp29 *inErr(20, tid, it);

    outErr(0, tid, it) = temp00 + temp02*errorProp2 + temp03*errorProp3 + temp04*errorProp4 + temp05*errorProp5;
    outErr(1, tid, it) = temp06 + temp08*errorProp2 + temp09*errorProp3 + temp10*errorProp4 + temp11*errorProp5;
    outErr(2, tid, it) = temp07 + temp08*errorProp8 + temp09*errorProp9 + temp10*errorProp10 + temp11*errorProp11;
    outErr(3, tid, it) = 0.0f;
    outErr(4, tid, it) = 0.0f;
    outErr(5, tid, it) = 0.0f;
    outErr(6, tid, it) = inErr(6, tid, it) + inErr(8, tid, it)*errorProp2 + inErr(9, tid, it)*errorProp3 + inErr(13, tid, it)*errorProp4 + inErr(18, tid, it)*errorProp5;
    outErr(7, tid, it) = inErr(7, tid, it) + inErr(8, tid, it)*errorProp8 + inErr(9, tid, it)*errorProp9 + inErr(13, tid, it)*errorProp10 + inErr(18, tid, it)*errorProp11;
    outErr(8, tid, it) = 0.0f;
    outErr(9, tid, it) = inErr(9, tid, it);
    outErr(10, tid, it) = temp24 + temp26*errorProp2 + temp27*errorProp3 + temp28*errorProp4 + temp29*errorProp5;
    outErr(11, tid, it) = temp25 + temp26*errorProp8 + temp27*errorProp9 + temp28*errorProp10 + temp29*errorProp11;
    outErr(12, tid, it) = 0.0f;
    outErr(13, tid, it) = temp27;
    outErr(14, tid, it) = temp26*errorProp26 + temp27*errorProp27 + temp28 + temp29*errorProp29;
    outErr(15, tid, it) = inErr(15, tid, it) + inErr(17, tid, it)*errorProp2 + inErr(18, tid, it)*errorProp3 + inErr(19, tid, it)*errorProp4 + inErr(20, tid, it)*errorProp5;
    outErr(16, tid, it) = inErr(16, tid, it) + inErr(17, tid, it)*errorProp8 + inErr(18, tid, it)*errorProp9 + inErr(19, tid, it)*errorProp10 + inErr(20, tid, it)*errorProp11;
    outErr(17, tid, it) = 0.0f;
    outErr(18, tid, it) = inErr(18, tid, it);
    outErr(19, tid, it) = inErr(17, tid, it)*errorProp26 + inErr(18, tid, it)*errorProp27 + inErr(19, tid, it) + inErr(20, tid, it)*errorProp29;
    outErr(20, tid, it) = inErr(20, tid, it);
  }

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
   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66[0]],
	                                       inputtrk.cov[SymOffsets66[(1*6+1)]],
	                                       inputtrk.cov[SymOffsets66[(2*6+2)]]);
   printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);

   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;
#if defined(__NVCOMPILER_CUDA__)
   constexpr auto order = FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER;
#else
   constexpr auto order = FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER;
#endif
   using MPTRKAccessorTp = MPTRKAccessor<order>;
   using MPHITAccessorTp = MPHITAccessor<order>;

   MPHIT_* hit    = prepareHits(inputhit);
   MPTRK_* outtrk = (MPTRK_*) malloc(nevts*nb*sizeof(MPTRK_));

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   auto trkNPtr = prepareTracksN<order>(inputtrk);
   std::unique_ptr<MPTRKAccessorTp> trkNaccPtr(new MPTRKAccessorTp(*trkNPtr));

   auto hitNPtr = prepareHitsN<order>(inputhit);
   std::unique_ptr<MPHITAccessorTp> hitNaccPtr(new MPHITAccessorTp(*hitNPtr));

   std::unique_ptr<MPTRK> outtrkNPtr(new MPTRK(ntrks, nevts));
   std::unique_ptr<MPTRKAccessorTp> outtrkNaccPtr(new MPTRKAccessorTp(*outtrkNPtr));

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   auto wall_start = std::chrono::high_resolution_clock::now();

   auto policy = std::execution::par_unseq;

   for(itr=0; itr<NITER; itr++) {

     const int outer_loop_range = nevts*nb;

     std::for_each(policy,
                   counting_iterator(0),
                   counting_iterator(outer_loop_range),
                   [=,&trkNacc = *trkNaccPtr,
                      &hitNacc = *hitNaccPtr,
                      &outtrkNacc = *outtrkNaccPtr] (const auto i) {
                     for(int layer=0; layer<nlayer; ++layer) {
                       propagateToZ<MPTRKAccessorTp, MPHITAccessorTp, bsize>(outtrkNacc, trkNacc, hitNacc, i, layer);
                       KalmanUpdate<MPTRKAccessorTp, MPHITAccessorTp, bsize>(outtrkNacc, hitNacc, i, layer);
                     }

                   });
#if defined(__NVCOMPILER_CUDA__) //artificial refresh pstl containers
      convertTracks<order>(outtrk, outtrkNPtr.get());
#endif

   } //end of itr loop

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   convertTracks<order>(outtrk, outtrkNPtr.get());
   convertHits<order>(hit, hitNPtr.get());

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
