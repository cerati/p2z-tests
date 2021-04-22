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
   static constexpr int n   = MPNTp::N;//matrix linear dim (total number of els)

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

   T& operator()(const int mat_idx, const int trkev_idx, const int b_idx, const int layer_idx) const {
     if      constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return data_[mat_idx*stride + layer_idx*NevtsNtbBsz + trkev_idx*bsz + b_idx];//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return data_[layer_idx*stride + mat_idx*NevtsNtbBsz + trkev_idx*bsz + b_idx];
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return data_[trkev_idx*stride+layer_idx*n*bsz+mat_idx*bsz+b_idx];
   }//i is the internal dof index

   T& operator()(const int thrd_idx, const int stride, const int blk_offset) const { return data_[thrd_idx*stride + blk_offset];}//

   int GetThreadOffset(const int thrd_idx, const int layer_idx = 0) const {
     if      constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return (layer_idx*NevtsNtbBsz + thrd_idx*bsz);//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return (layer_idx*stride + thrd_idx*bsz);
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return (thrd_idx*stride+layer_idx*n*bsz);
   }

   int GetThreadStride() const {
     if      constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return stride;//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return NevtsNtbBsz;
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return bsz;
   }

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
          rA->par(ip, tid, it, 0) = (1+smear*randn(0,1))*inputtrk.par[ip];
    	  }
    	  //cov
    	  for (size_t ip=0;ip<21;++ip) {
          rA->cov(ip, tid, it, 0) = (1+smear*randn(0,1))*inputtrk.cov[ip];
    	  }
    	  //q
        rA->q(0, tid, it, 0) = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);
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



//Pure static version:

template <typename T, int N, int bSize>
struct MPNX_ {
   std::array<T,N*bSize> data;
   //basic accessors
   const T& operator[](const int idx) const {return data[idx];}
   T& operator[](const int idx) {return data[idx];}
};

using MP1I_    = MPNX_<int,   1 , bsize>;
using MP3F_    = MPNX_<float, 3 , bsize>;
using MP6F_    = MPNX_<float, 6 , bsize>;
using MP3x3SF_ = MPNX_<float, 6 , bsize>;
using MP6x6SF_ = MPNX_<float, 21, bsize>;
using MP6x6F_  = MPNX_<float, 36, bsize>;
using MP3x3_   = MPNX_<float, 9 , bsize>;
using MP3x6_   = MPNX_<float, 18, bsize>;

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

// HOST methods and routines

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
    	    out[tid].par.data[it + ip*bsize] = inpA->par(ip, tid, it, 0);
    	  }
    	  //cov
    	  for (size_t ip=0;ip<21;++ip) {
    	    out[tid].cov.data[it + ip*bsize] = inpA->cov(ip, tid, it, 0);
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

//Main (sub)routines

auto PosInMtrx = [](const size_t &&i, const size_t &&j, const size_t &&D, const size_t block_size = 1) constexpr {return block_size*(i*D+j);};

template<typename MP6x6SFAccessor_, size_t block_size = 1>
inline void MultHelixPropEndcap(const MP6x6F_ &a, const MP6x6SFAccessor_ &b, MP6x6F_ &c, const int tid) {

  const auto stride = b.GetThreadStride();
  const auto offset = b.GetThreadOffset(tid);

  #pragma simd
  for (int it = 0;it < block_size; it++) {

    const auto blk_offset = offset+it;

    c[ 0*block_size + it] = b(0, stride, blk_offset) + a[PosInMtrx(0,2,6, block_size) + it] *b(3, stride, blk_offset) + a[PosInMtrx(0,3,6, block_size) + it] *b(6, stride, blk_offset) + a[PosInMtrx(0,4,6, block_size) + it]  *b(10,stride, blk_offset) + a[PosInMtrx(0,5,6, block_size) + it]  *b(15,stride, blk_offset);
    c[ 1*block_size + it] = b(1, stride, blk_offset) + a[PosInMtrx(0,2,6, block_size) + it] *b(4, stride, blk_offset) + a[PosInMtrx(0,3,6, block_size) + it] *b(7, stride, blk_offset) + a[PosInMtrx(0,4,6, block_size) + it]  *b(11,stride, blk_offset) + a[PosInMtrx(0,5,6, block_size) + it]  *b(16,stride, blk_offset);
    c[ 2*block_size + it] = b(3, stride, blk_offset) + a[PosInMtrx(0,2,6, block_size) + it] *b(5, stride, blk_offset) + a[PosInMtrx(0,3,6, block_size) + it] *b(8, stride, blk_offset) + a[PosInMtrx(0,4,6, block_size) + it]  *b(12,stride, blk_offset) + a[PosInMtrx(0,5,6, block_size) + it]  *b(17,stride, blk_offset);
    c[ 3*block_size + it] = b(6, stride, blk_offset) + a[PosInMtrx(0,2,6, block_size) + it] *b(8, stride, blk_offset) + a[PosInMtrx(0,3,6, block_size) + it] *b(9, stride, blk_offset) + a[PosInMtrx(0,4,6, block_size) + it]  *b(13,stride, blk_offset) + a[PosInMtrx(0,5,6, block_size) + it]  *b(18,stride, blk_offset);
    c[ 4*block_size + it] = b(10,stride, blk_offset) + a[PosInMtrx(0,2,6, block_size) + it] *b(12,stride, blk_offset) + a[PosInMtrx(0,3,6, block_size) + it] *b(13,stride, blk_offset) + a[PosInMtrx(0,4,6, block_size) + it]  *b(14,stride, blk_offset) + a[PosInMtrx(0,5,6, block_size) + it]  *b(19,stride, blk_offset);
    c[ 5*block_size + it] = b(15,stride, blk_offset) + a[PosInMtrx(0,2,6, block_size) + it] *b(17,stride, blk_offset) + a[PosInMtrx(0,3,6, block_size) + it] *b(18,stride, blk_offset) + a[PosInMtrx(0,4,6, block_size) + it]  *b(19,stride, blk_offset) + a[PosInMtrx(0,5,6, block_size) + it]  *b(20,stride, blk_offset);
    c[ 6*block_size + it] = b(1, stride, blk_offset) + a[PosInMtrx(1,2,6, block_size) + it] *b(3, stride, blk_offset) + a[PosInMtrx(1,3,6, block_size) + it] *b(6, stride, blk_offset) + a[PosInMtrx(1,4,6, block_size) + it] *b(10,stride, blk_offset) + a[PosInMtrx(1,5,6, block_size) + it] *b(15,stride, blk_offset);
    c[ 7*block_size + it] = b(2, stride, blk_offset) + a[PosInMtrx(1,2,6, block_size) + it] *b(4, stride, blk_offset) + a[PosInMtrx(1,3,6, block_size) + it] *b(7, stride, blk_offset) + a[PosInMtrx(1,4,6, block_size) + it] *b(11,stride, blk_offset) + a[PosInMtrx(1,5,6, block_size) + it] *b(16,stride, blk_offset);
    c[ 8*block_size + it] = b(4, stride, blk_offset) + a[PosInMtrx(1,2,6, block_size) + it] *b(5, stride, blk_offset) + a[PosInMtrx(1,3,6, block_size) + it] *b(8, stride, blk_offset) + a[PosInMtrx(1,4,6, block_size) + it] *b(12,stride, blk_offset) + a[PosInMtrx(1,5,6, block_size) + it] *b(17,stride, blk_offset);
    c[ 9*block_size + it] = b(7, stride, blk_offset) + a[PosInMtrx(1,2,6, block_size) + it] *b(8, stride, blk_offset) + a[PosInMtrx(1,3,6, block_size) + it] *b(9, stride, blk_offset) + a[PosInMtrx(1,4,6, block_size) + it] *b(13,stride, blk_offset) + a[PosInMtrx(1,5,6, block_size) + it] *b(18,stride, blk_offset);
    c[10*block_size + it] = b(11,stride, blk_offset) + a[PosInMtrx(1,2,6, block_size) + it] *b(12,stride, blk_offset) + a[PosInMtrx(1,3,6, block_size) + it] *b(13,stride, blk_offset) + a[PosInMtrx(1,4,6, block_size) + it] *b(14,stride, blk_offset) + a[PosInMtrx(1,5,6, block_size) + it] *b(19,stride, blk_offset);
    c[11*block_size + it] = b(16,stride, blk_offset) + a[PosInMtrx(1,2,6, block_size) + it] *b(17,stride, blk_offset) + a[PosInMtrx(1,3,6, block_size) + it] *b(18,stride, blk_offset) + a[PosInMtrx(1,4,6, block_size) + it] *b(19,stride, blk_offset) + a[PosInMtrx(1,5,6, block_size) + it] *b(20,stride, blk_offset);

    c[12*block_size + it] = 0.0f;
    c[13*block_size + it] = 0.0f;
    c[14*block_size + it] = 0.0f;
    c[15*block_size + it] = 0.0f;
    c[16*block_size + it] = 0.0f;
    c[17*block_size + it] = 0.0f;
    c[18*block_size + it] = b(6,stride, blk_offset);
    c[19*block_size + it] = b(7,stride, blk_offset);
    c[20*block_size + it] = b(8,stride, blk_offset);
    c[21*block_size + it] = b(9,stride, blk_offset);
    c[22*block_size + it] = b(13,stride, blk_offset);
    c[23*block_size + it] = b(18,stride, blk_offset);


    c[24*block_size + it] = a[PosInMtrx(4,2,6, block_size) + it] *b(3, stride, blk_offset) + a[PosInMtrx(4,3,6, block_size) + it] *b(6, stride, blk_offset) + b(10, stride, blk_offset) + a[PosInMtrx(4,5,6, block_size) + it] *b(15, stride, blk_offset);
    c[25*block_size + it] = a[PosInMtrx(4,2,6, block_size) + it] *b(4, stride, blk_offset) + a[PosInMtrx(4,3,6, block_size) + it] *b(7, stride, blk_offset) + b(11, stride, blk_offset) + a[PosInMtrx(4,5,6, block_size) + it] *b(16, stride, blk_offset);
    c[26*block_size + it] = a[PosInMtrx(4,2,6, block_size) + it] *b(5, stride, blk_offset) + a[PosInMtrx(4,3,6, block_size) + it] *b(8, stride, blk_offset) + b(12, stride, blk_offset) + a[PosInMtrx(4,5,6, block_size) + it] *b(17, stride, blk_offset);
    c[27*block_size + it] = a[PosInMtrx(4,2,6, block_size) + it] *b(8, stride, blk_offset) + a[PosInMtrx(4,3,6, block_size) + it] *b(9, stride, blk_offset) + b(13, stride, blk_offset) + a[PosInMtrx(4,5,6, block_size) + it] *b(18, stride, blk_offset);
    c[28*block_size + it] = a[PosInMtrx(4,2,6, block_size) + it] *b(12, stride, blk_offset) + a[PosInMtrx(4,3,6, block_size) + it] *b(13, stride, blk_offset) + b(14, stride, blk_offset) + a[PosInMtrx(4,5,6, block_size) + it] *b(19, stride, blk_offset);
    c[29*block_size + it] = a[PosInMtrx(4,2,6, block_size) + it] *b(17, stride, blk_offset) + a[PosInMtrx(4,3,6, block_size) + it] *b(18, stride, blk_offset) + b(19, stride, blk_offset) + a[PosInMtrx(4,5,6, block_size) + it] *b(20, stride, blk_offset);

    c[30*block_size + it] = b(15,stride, blk_offset);
    c[31*block_size + it] = b(16,stride, blk_offset);
    c[32*block_size + it] = b(17,stride, blk_offset);
    c[33*block_size + it] = b(18,stride, blk_offset);
    c[34*block_size + it] = b(19,stride, blk_offset);
    c[35*block_size + it] = b(20,stride, blk_offset);
  }
  return;
}

template<typename MP6x6SFAccessor_, size_t block_size = 1>
inline void MultHelixPropTranspEndcap(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SFAccessor_ &c, const int tid) {

  const auto stride = c.GetThreadStride();
  const auto offset = c.GetThreadOffset(tid);

  #pragma simd
  for (int it = 0;it < block_size; it++) {
    const auto blk_offset = offset+it;

    c( 0, stride, blk_offset) = b[0*block_size + it] + b[2*block_size + it]*a[PosInMtrx(0,2,6, block_size) + it] + b[3*block_size + it]*a[PosInMtrx(0,3,6, block_size) + it] + b[4*block_size + it]*a[PosInMtrx(0,4,6, block_size) + it] + b[5*block_size + it]*a[PosInMtrx(0,5,6, block_size) + it];
    c( 1, stride, blk_offset) = b[6*block_size + it] + b[8*block_size + it]*a[PosInMtrx(0,2,6, block_size) + it] + b[9*block_size + it]*a[PosInMtrx(0,3,6, block_size) + it] + b[10*block_size + it]*a[PosInMtrx(0,4,6, block_size) + it] + b[11*block_size + it]*a[PosInMtrx(0,5,6, block_size) + it];
    c( 2, stride, blk_offset) = b[7*block_size + it] + b[8*block_size + it]*a[PosInMtrx(1,2,6, block_size) + it] + b[9*block_size + it]*a[PosInMtrx(1,3,6, block_size) + it] + b[10*block_size + it]*a[PosInMtrx(1,4,6, block_size) + it] + b[11*block_size + it]*a[PosInMtrx(1,5,6, block_size) + it];
    c( 3, stride, blk_offset) = 0.0f;
    c( 4, stride, blk_offset) = 0.0f;
    c( 5, stride, blk_offset) = 0.0f;
    c( 6, stride, blk_offset) = b[18*block_size + it] + b[20*block_size + it]*a[PosInMtrx(0,2,6, block_size) + it] + b[21*block_size + it]*a[PosInMtrx(0,3,6, block_size) + it] + b[22*block_size + it]*a[PosInMtrx(0,4,6, block_size) + it] + b[23*block_size + it]*a[PosInMtrx(0,5,6, block_size) + it];
    c( 7, stride, blk_offset) = b[19*block_size + it] + b[20*block_size + it]*a[PosInMtrx(1,2,6, block_size) + it] + b[21*block_size + it]*a[PosInMtrx(1,3,6, block_size) + it] + b[22*block_size + it]*a[PosInMtrx(1,4,6, block_size) + it] + b[23*block_size + it]*a[PosInMtrx(1,5,6, block_size) + it];
    c( 8, stride, blk_offset) = 0.0f;
    c( 9, stride, blk_offset) = b[21*block_size + it];
    c(10, stride, blk_offset) = b[24*block_size + it] + b[26*block_size + it]*a[PosInMtrx(0,2,6, block_size) + it] + b[27*block_size + it]*a[PosInMtrx(0,3,6, block_size) + it] + b[28*block_size + it]*a[PosInMtrx(0,4,6, block_size) + it] + b[29*block_size + it]*a[PosInMtrx(0,5,6, block_size) + it];
    c(11, stride, blk_offset) = b[25*block_size + it] + b[26*block_size + it]*a[PosInMtrx(1,2,6, block_size) + it] + b[27*block_size + it]*a[PosInMtrx(1,3,6, block_size) + it] + b[28*block_size + it]*a[PosInMtrx(1,4,6, block_size) + it] + b[29*block_size + it]*a[PosInMtrx(1,5,6, block_size) + it];
    c(12, stride, blk_offset) = 0.0f;
    c(13, stride, blk_offset) = b[27*block_size + it];
    c(14, stride, blk_offset) = b[26*block_size + it]*a[PosInMtrx(4,2,6, block_size) + it] + b[27*block_size + it]*a[PosInMtrx(4,3,6, block_size) + it] + b[28*block_size + it] + b[29*block_size + it]*a[PosInMtrx(4,5,6, block_size) + it];
    c(15, stride, blk_offset) = b[30*block_size + it] + b[32*block_size + it]*a[PosInMtrx(0,2,6, block_size) + it] + b[33*block_size + it]*a[PosInMtrx(0,3,6, block_size) + it] + b[34*block_size + it]*a[PosInMtrx(0,4,6, block_size) + it] + b[35*block_size + it]*a[PosInMtrx(0,5,6, block_size) + it];
    c(16, stride, blk_offset) = b[31*block_size + it] + b[32*block_size + it]*a[PosInMtrx(1,2,6, block_size) + it] + b[33*block_size + it]*a[PosInMtrx(1,3,6, block_size) + it] + b[34*block_size + it]*a[PosInMtrx(1,4,6, block_size) + it] + b[35*block_size + it]*a[PosInMtrx(1,5,6, block_size) + it];
    c(17, stride, blk_offset) = 0.0f;
    c(18, stride, blk_offset) = b[33*block_size + it];
    c(19, stride, blk_offset) = b[32*block_size + it]*a[PosInMtrx(4,2,6, block_size) + it] + b[33*block_size + it]*a[PosInMtrx(4,3,6, block_size) + it] + b[34*block_size + it] + b[35*block_size + it]*a[PosInMtrx(4,5,6, block_size) + it];
    c(20, stride, blk_offset) = b[35*block_size + it];

  }
  return;
}

template<typename AccessorTp1, typename AccessorTp2, size_t block_size = 1>
inline void KalmanGainInv(const AccessorTp1 &a, const AccessorTp2 &b, MP3x3_ &c, const int tid, const int lay) {
  const auto a_stride  = a.GetThreadStride();
  const auto a_offset_ = a.GetThreadOffset(tid);

  const auto b_stride  = b.GetThreadStride();
  const auto b_offset_ = b.GetThreadOffset(tid, lay);
#pragma omp simd
  for (int it = 0; it < block_size; ++it)
  {
    const auto a_offset = a_offset_+it;
    const auto b_offset = b_offset_+it;

    double det =
        ((a(0, a_stride, a_offset)+b(0, b_stride, b_offset))*(((a(6, a_stride, a_offset)+b(3, b_stride, b_offset)) *(a(11,a_stride, a_offset)+b(5, b_stride, b_offset))) - ((a(7, a_stride, a_offset)+b(4, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))))) -
        ((a(1, a_stride, a_offset)+b(1, b_stride, b_offset))*(((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(11,a_stride, a_offset)+b(5, b_stride, b_offset))) - ((a(7, a_stride, a_offset)+b(4, b_stride, b_offset)) *(a(2, a_stride, a_offset)+b(2, b_stride, b_offset))))) +
        ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset))*(((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(6, a_stride, a_offset)+b(3, b_stride, b_offset)))));

    float invdet = 1.0 / det;

    c[0*block_size+it] =   invdet*(((a(6, a_stride, a_offset)+b(3, b_stride, b_offset)) *(a(11,a_stride, a_offset)+b(5, b_stride, b_offset))) - ((a(7, a_stride, a_offset)+b(4, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))));
    c[1*block_size+it] =  -invdet*(((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(11,a_stride, a_offset)+b(5, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))));
    c[2*block_size+it] =   invdet*(((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))));
    c[3*block_size+it] =  -invdet*(((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(11,a_stride, a_offset)+b(5, b_stride, b_offset))) - ((a(7, a_stride, a_offset)+b(4, b_stride, b_offset)) *(a(2, a_stride, a_offset)+b(2, b_stride, b_offset))));
    c[4*block_size+it] =   invdet*(((a(0, a_stride, a_offset)+b(0, b_stride, b_offset)) *(a(11,a_stride, a_offset)+b(5, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(2, a_stride, a_offset)+b(2, b_stride, b_offset))));
    c[5*block_size+it] =  -invdet*(((a(0, a_stride, a_offset)+b(0, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(1, a_stride, a_offset)+b(1, b_stride, b_offset))));
    c[6*block_size+it] =   invdet*(((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(6, a_stride, a_offset)+b(3, b_stride, b_offset))));
    c[7*block_size+it] =  -invdet*(((a(0, a_stride, a_offset)+b(0, b_stride, b_offset)) *(a(7, a_stride, a_offset)+b(4, b_stride, b_offset))) - ((a(2, a_stride, a_offset)+b(2, b_stride, b_offset)) *(a(1, a_stride, a_offset)+b(1, b_stride, b_offset))));
    c[8*block_size+it] =   invdet*(((a(0, a_stride, a_offset)+b(0, b_stride, b_offset)) *(a(6, a_stride, a_offset)+b(3, b_stride, b_offset))) - ((a(1, a_stride, a_offset)+b(1, b_stride, b_offset)) *(a(1, a_stride, a_offset)+b(1, b_stride, b_offset))));

  }
}

template<typename AccessorTp, size_t block_size = 1>
inline void KalmanGain(const AccessorTp &a, const MP3x3_ &b, MP3x6_ &c, const int tid) {
  const auto a_stride = a.GetThreadStride();
  const auto a_offset_= a.GetThreadOffset(tid);
#pragma simd
  for (int it = 0; it < block_size; ++it)
  {
    const auto a_offset = a_offset_+it;

    c[ 0*block_size+it] = a(0, a_stride, a_offset)*b[0*block_size+it] + a( 1, a_stride, a_offset)*b[3*block_size+it] + a( 2, a_stride, a_offset)*b[6*block_size+it];
    c[ 1*block_size+it] = a(0, a_stride, a_offset)*b[1*block_size+it] + a( 1, a_stride, a_offset)*b[4*block_size+it] + a( 2, a_stride, a_offset)*b[7*block_size+it];
    c[ 2*block_size+it] = a(0, a_stride, a_offset)*b[2*block_size+it] + a( 1, a_stride, a_offset)*b[5*block_size+it] + a( 2, a_stride, a_offset)*b[8*block_size+it];
    c[ 3*block_size+it] = a(1, a_stride, a_offset)*b[0*block_size+it] + a( 6, a_stride, a_offset)*b[3*block_size+it] + a( 7, a_stride, a_offset)*b[6*block_size+it];
    c[ 4*block_size+it] = a(1, a_stride, a_offset)*b[1*block_size+it] + a( 6, a_stride, a_offset)*b[4*block_size+it] + a( 7, a_stride, a_offset)*b[7*block_size+it];
    c[ 5*block_size+it] = a(1, a_stride, a_offset)*b[2*block_size+it] + a( 6, a_stride, a_offset)*b[5*block_size+it] + a( 7, a_stride, a_offset)*b[8*block_size+it];
    c[ 6*block_size+it] = a(2, a_stride, a_offset)*b[0*block_size+it] + a( 7, a_stride, a_offset)*b[3*block_size+it] + a(11, a_stride, a_offset)*b[6*block_size+it];
    c[ 7*block_size+it] = a(2, a_stride, a_offset)*b[1*block_size+it] + a( 7, a_stride, a_offset)*b[4*block_size+it] + a(11, a_stride, a_offset)*b[7*block_size+it];
    c[ 8*block_size+it] = a(2, a_stride, a_offset)*b[2*block_size+it] + a( 7, a_stride, a_offset)*b[5*block_size+it] + a(11, a_stride, a_offset)*b[8*block_size+it];
    c[ 9*block_size+it] = a(3, a_stride, a_offset)*b[0*block_size+it] + a( 8, a_stride, a_offset)*b[3*block_size+it] + a(12, a_stride, a_offset)*b[6*block_size+it];
    c[10*block_size+it] = a(3, a_stride, a_offset)*b[1*block_size+it] + a( 8, a_stride, a_offset)*b[4*block_size+it] + a(12, a_stride, a_offset)*b[7*block_size+it];
    c[11*block_size+it] = a(3, a_stride, a_offset)*b[2*block_size+it] + a( 8, a_stride, a_offset)*b[5*block_size+it] + a(12, a_stride, a_offset)*b[8*block_size+it];
    c[12*block_size+it] = a(4, a_stride, a_offset)*b[0*block_size+it] + a( 9, a_stride, a_offset)*b[3*block_size+it] + a(13, a_stride, a_offset)*b[6*block_size+it];
    c[13*block_size+it] = a(4, a_stride, a_offset)*b[1*block_size+it] + a( 9, a_stride, a_offset)*b[4*block_size+it] + a(13, a_stride, a_offset)*b[7*block_size+it];
    c[14*block_size+it] = a(4, a_stride, a_offset)*b[2*block_size+it] + a( 9, a_stride, a_offset)*b[5*block_size+it] + a(13, a_stride, a_offset)*b[8*block_size+it];
    c[15*block_size+it] = a(5, a_stride, a_offset)*b[0*block_size+it] + a(10, a_stride, a_offset)*b[3*block_size+it] + a(14, a_stride, a_offset)*b[6*block_size+it];
    c[16*block_size+it] = a(5, a_stride, a_offset)*b[1*block_size+it] + a(10, a_stride, a_offset)*b[4*block_size+it] + a(14, a_stride, a_offset)*b[7*block_size+it];
    c[17*block_size+it] = a(5, a_stride, a_offset)*b[2*block_size+it] + a(10, a_stride, a_offset)*b[5*block_size+it] + a(14, a_stride, a_offset)*b[8*block_size+it];

  }
}

template <class MPTRKAccessors, class MPHITAccessors, size_t block_size = 1>
void KalmanUpdate(MPTRKAccessors       &obtracks,
		              const MPHITAccessors &bhits,
                  const int tid,
                  const int lay) {
  using MP6Faccessor    = typename MPTRKAccessors::MP6FAccessor;
  using MP6x6SFaccessor = typename MPTRKAccessors::MP6x6SFAccessor;
  using MP3x3SFaccessor = typename MPHITAccessors::MP3x3SFAccessor;
  using MP3Faccessor    = typename MPHITAccessors::MP3FAccessor;

  const MP3x3SFaccessor &hitErr   = bhits.cov;
  const MP3Faccessor    &msP      = bhits.pos;

  MP6x6SFaccessor  &trkErr = obtracks.cov;
  MP6Faccessor     &inPar  = obtracks.par;

  const auto terr_stride = trkErr.GetThreadStride();
  const auto terr_offset = trkErr.GetThreadOffset(tid);

  const auto ipar_stride = inPar.GetThreadStride();
  const auto ipar_offset = inPar.GetThreadOffset(tid);

  MP3x3_ temp;//thread private data?
  MP3x6_ kGain;//thread private data?
 
  KalmanGainInv<MP6x6SFaccessor, MP3x3SFaccessor, block_size>(trkErr, hitErr, temp, tid, lay);
  KalmanGain<MP6x6SFaccessor, block_size>(trkErr, temp, kGain, tid);

#pragma simd
  for (int it = 0; it < block_size; it++) {
    const auto terr_blk_offset = terr_offset+it;
    const auto ipar_blk_offset = ipar_offset+it;

    const auto xin     = inPar(iparX, ipar_stride, ipar_blk_offset);
    const auto yin     = inPar(iparY, ipar_stride, ipar_blk_offset);
    const auto zin     = inPar(iparZ, ipar_stride, ipar_blk_offset);
    const auto ptin    = 1. / inPar(iparIpt, ipar_stride, ipar_blk_offset);
    const auto phiin   = inPar(iparPhi, ipar_stride, ipar_blk_offset);
    const auto thetain = inPar(iparTheta, ipar_stride, ipar_blk_offset);
    const auto xout    = msP(iparX, tid, it, lay);
    const auto yout    = msP(iparY, tid, it, lay);

    const auto xnew     = xin     + (kGain[ 0*block_size+it]*(xout-xin)) +(kGain[ 1*block_size+it]*(yout-yin));
    const auto ynew     = yin     + (kGain[ 3*block_size+it]*(xout-xin)) +(kGain[ 4*block_size+it]*(yout-yin));
    const auto znew     = zin     + (kGain[ 6*block_size+it]*(xout-xin)) +(kGain[ 7*block_size+it]*(yout-yin));
    const auto ptnew    = ptin    + (kGain[ 9*block_size+it]*(xout-xin)) +(kGain[10*block_size+it]*(yout-yin));
    const auto phinew   = phiin   + (kGain[12*block_size+it]*(xout-xin)) +(kGain[13*block_size+it]*(yout-yin));
    const auto thetanew = thetain + (kGain[15*block_size+it]*(xout-xin)) +(kGain[16*block_size+it]*(yout-yin));

    //
    trkErr(0, terr_stride, terr_blk_offset) = trkErr(0, terr_stride, terr_blk_offset) - (kGain[ 0*block_size+it]*trkErr(0, terr_stride, terr_blk_offset)+kGain[ 1*block_size+it]*trkErr(1, terr_stride, terr_blk_offset)+kGain[ 2*block_size+it]*trkErr(2, terr_stride, terr_blk_offset));
    //

    temp[0*block_size+it] = trkErr(6, terr_stride, terr_blk_offset) - (kGain[ 3*block_size+it]*trkErr(1, terr_stride, terr_blk_offset)+kGain[ 4*block_size+it]*trkErr(6, terr_stride, terr_blk_offset)+kGain[ 5*block_size+it]*trkErr(7, terr_stride, terr_blk_offset));
    //*
    trkErr(1, terr_stride, terr_blk_offset) = trkErr(1, terr_stride, terr_blk_offset) - (kGain[ 0*block_size+it]*trkErr(1, terr_stride, terr_blk_offset)+kGain[ 1*block_size+it]*trkErr(6, terr_stride, terr_blk_offset)+kGain[ 2*block_size+it]*trkErr(7, terr_stride, terr_blk_offset));
    //*
    trkErr(6, terr_stride, terr_blk_offset) = temp[0*block_size+it];

    temp[0*block_size+it] = trkErr(2, terr_stride, terr_blk_offset) - (kGain[ 0*block_size+it]*trkErr(2, terr_stride, terr_blk_offset)+kGain[ 1*block_size+it]*trkErr(7, terr_stride, terr_blk_offset)+kGain[ 2*block_size+it]*trkErr(11, terr_stride, terr_blk_offset));
    temp[1*block_size+it] = trkErr(7, terr_stride, terr_blk_offset) - (kGain[ 3*block_size+it]*trkErr(2, terr_stride, terr_blk_offset)+kGain[ 4*block_size+it]*trkErr(7, terr_stride, terr_blk_offset)+kGain[ 5*block_size+it]*trkErr(11, terr_stride, terr_blk_offset));
    //**
    trkErr(11, terr_stride, terr_blk_offset) = trkErr(11, terr_stride, terr_blk_offset) - (kGain[ 6*block_size+it]*trkErr(2, terr_stride, terr_blk_offset)+kGain[ 7*block_size+it]*trkErr(7, terr_stride, terr_blk_offset)+kGain[ 8*block_size+it]*trkErr(11, terr_stride, terr_blk_offset));
    //**
    trkErr(2, terr_stride, terr_blk_offset)  = temp[0*block_size+it];
    trkErr(7, terr_stride, terr_blk_offset)  = temp[1*block_size+it];

    temp[0*block_size+it] = trkErr( 3, terr_stride, terr_blk_offset) - (kGain[ 0*block_size+it]*trkErr(3, terr_stride, terr_blk_offset)+kGain[ 1*block_size+it]*trkErr(8, terr_stride, terr_blk_offset)+kGain[ 2*block_size+it]*trkErr(12, terr_stride, terr_blk_offset));
    temp[1*block_size+it] = trkErr(12, terr_stride, terr_blk_offset) - (kGain[ 6*block_size+it]*trkErr(3, terr_stride, terr_blk_offset)+kGain[ 7*block_size+it]*trkErr(8, terr_stride, terr_blk_offset)+kGain[ 8*block_size+it]*trkErr(12, terr_stride, terr_blk_offset));
    //***
    trkErr(15, terr_stride, terr_blk_offset) = trkErr(15, terr_stride, terr_blk_offset) - (kGain[ 9*block_size+it]*trkErr(3, terr_stride, terr_blk_offset)+kGain[10*block_size+it]*trkErr(8, terr_stride, terr_blk_offset)+kGain[11*block_size+it]*trkErr(12, terr_stride, terr_blk_offset));
    trkErr( 8, terr_stride, terr_blk_offset) = trkErr( 8, terr_stride, terr_blk_offset) - (kGain[ 3*block_size+it]*trkErr(3, terr_stride, terr_blk_offset)+kGain[ 4*block_size+it]*trkErr(8, terr_stride, terr_blk_offset)+kGain[ 5*block_size+it]*trkErr(12, terr_stride, terr_blk_offset));
    //***
    trkErr( 3, terr_stride, terr_blk_offset) = temp[0*block_size+it];
    trkErr(12, terr_stride, terr_blk_offset) = temp[1*block_size+it];

    temp[0*block_size+it] = trkErr( 9, terr_stride, terr_blk_offset) - (kGain[ 3*block_size+it]*trkErr(4, terr_stride, terr_blk_offset)+kGain[ 4*block_size+it]*trkErr(9, terr_stride, terr_blk_offset)+kGain[ 5*block_size+it]*trkErr(13, terr_stride, terr_blk_offset));
    temp[1*block_size+it] = trkErr(13, terr_stride, terr_blk_offset) - (kGain[ 6*block_size+it]*trkErr(4, terr_stride, terr_blk_offset)+kGain[ 7*block_size+it]*trkErr(9, terr_stride, terr_blk_offset)+kGain[ 8*block_size+it]*trkErr(13, terr_stride, terr_blk_offset));
    //****
    trkErr(16, terr_stride, terr_blk_offset) = trkErr(16, terr_stride, terr_blk_offset) - (kGain[ 9*block_size+it]*trkErr(4, terr_stride, terr_blk_offset)+kGain[10*block_size+it]*trkErr(9, terr_stride, terr_blk_offset)+kGain[11*block_size+it]*trkErr(13, terr_stride, terr_blk_offset));
    trkErr(18, terr_stride, terr_blk_offset) = trkErr(18, terr_stride, terr_blk_offset) - (kGain[12*block_size+it]*trkErr(4, terr_stride, terr_blk_offset)+kGain[13*block_size+it]*trkErr(9, terr_stride, terr_blk_offset)+kGain[14*block_size+it]*trkErr(13, terr_stride, terr_blk_offset));
    trkErr( 4, terr_stride, terr_blk_offset) = trkErr( 4, terr_stride, terr_blk_offset) - (kGain[ 0*block_size+it]*trkErr(4, terr_stride, terr_blk_offset)+kGain[ 1*block_size+it]*trkErr(9, terr_stride, terr_blk_offset)+kGain[ 2*block_size+it]*trkErr(13, terr_stride, terr_blk_offset));
    //****
    trkErr( 9, terr_stride, terr_blk_offset) = temp[0*block_size+it];
    trkErr(13, terr_stride, terr_blk_offset) = temp[1*block_size+it];

    temp[0*block_size+it] = trkErr(10, terr_stride, terr_blk_offset) - (kGain[ 3*block_size+it]*trkErr(5, terr_stride, terr_blk_offset)+kGain[ 4*block_size+it]*trkErr(10, terr_stride, terr_blk_offset)+kGain[ 5*block_size+it]*trkErr(14, terr_stride, terr_blk_offset));
    temp[1*block_size+it] = trkErr(14, terr_stride, terr_blk_offset) - (kGain[ 6*block_size+it]*trkErr(5, terr_stride, terr_blk_offset)+kGain[ 7*block_size+it]*trkErr(10, terr_stride, terr_blk_offset)+kGain[ 8*block_size+it]*trkErr(14, terr_stride, terr_blk_offset));
    //*****
    trkErr(17, terr_stride, terr_blk_offset) = trkErr(17, terr_stride, terr_blk_offset) - (kGain[ 9*block_size+it]*trkErr(5, terr_stride, terr_blk_offset)+kGain[10*block_size+it]*trkErr(10, terr_stride, terr_blk_offset)+kGain[11*block_size+it]*trkErr(14, terr_stride, terr_blk_offset));
    trkErr(19, terr_stride, terr_blk_offset) = trkErr(19, terr_stride, terr_blk_offset) - (kGain[12*block_size+it]*trkErr(5, terr_stride, terr_blk_offset)+kGain[13*block_size+it]*trkErr(10, terr_stride, terr_blk_offset)+kGain[14*block_size+it]*trkErr(14, terr_stride, terr_blk_offset));
    trkErr(20, terr_stride, terr_blk_offset) = trkErr(20, terr_stride, terr_blk_offset) - (kGain[15*block_size+it]*trkErr(5, terr_stride, terr_blk_offset)+kGain[16*block_size+it]*trkErr(10, terr_stride, terr_blk_offset)+kGain[17*block_size+it]*trkErr(14, terr_stride, terr_blk_offset));
    trkErr( 5, terr_stride, terr_blk_offset) = trkErr( 5, terr_stride, terr_blk_offset) - (kGain[ 0*block_size+it]*trkErr(5, terr_stride, terr_blk_offset)+kGain[ 1*block_size+it]*trkErr(10, terr_stride, terr_blk_offset)+kGain[ 2*block_size+it]*trkErr(14, terr_stride, terr_blk_offset));
    //*****
    trkErr(10, terr_stride, terr_blk_offset) = temp[0*block_size+it];
    trkErr(14, terr_stride, terr_blk_offset) = temp[1*block_size+it];


    inPar(iparX,ipar_stride, ipar_blk_offset)     = xnew;
    inPar(iparY,ipar_stride, ipar_blk_offset)     = ynew;
    inPar(iparZ,ipar_stride, ipar_blk_offset)     = znew;
    inPar(iparIpt,ipar_stride, ipar_blk_offset)   = ptnew;
    inPar(iparPhi,ipar_stride, ipar_blk_offset)   = phinew;
    inPar(iparTheta,ipar_stride, ipar_blk_offset) = thetanew;
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

  using MP6Faccessor    = typename MPTRKAccessors::MP6FAccessor;
  using MP1Iaccessor    = typename MPTRKAccessors::MP1IAccessor;
  using MP6x6SFaccessor = typename MPTRKAccessors::MP6x6SFAccessor;
  using MP3Faccessor    = typename MPHITAccessors::MP3FAccessor;

  const MP6Faccessor &inPar    = btracks.par;
  const MP1Iaccessor &inChg    = btracks.q  ;
  const MP6x6SFaccessor &inErr = btracks.cov;

  const MP3Faccessor &msP      = bhits.pos;

  MP6x6SFaccessor &outErr    = obtracks.cov;
  MP6Faccessor    &outPar    = obtracks.par;

  const auto par_stride = inPar.GetThreadStride();
  const auto par_offset = inPar.GetThreadOffset(tid);

  MP6x6F_ temp;
  MP6x6F_ errorProp;

#pragma simd
  for (int it = 0;it < block_size; it++) {
    const auto par_blk_offset  = par_offset+it;

    const float zout = msP(iparZ, tid, it, lay);
    const float k    = inChg(0, tid, it, 0)*kfact;//100/3.8;
    const float deltaZ = zout - inPar(iparZ, par_stride, par_blk_offset);
    const float pt     = inPar(iparIpt, par_stride, par_blk_offset);
    const float cosP   = cosf(inPar(iparPhi, par_stride, par_blk_offset));//inPar(iparPhi, par_stride, par_blk_offset)
    const float sinP   = sinf(inPar(iparPhi, par_stride, par_blk_offset));
    const float cosT   = cosf(inPar(iparTheta, par_stride, par_blk_offset));
    const float sinT   = sinf(inPar(iparTheta, par_stride, par_blk_offset));

    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float icosT = 1.0/cosT;
    const float icosTk = icosT/k;
    const float alpha = deltaZ*sinT*pt*icosTk;

    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;

    outPar(iparX,par_stride, par_blk_offset) = inPar(iparX, par_stride, par_blk_offset) + k*(pxin*sina - pyin*(1.-cosa));
    outPar(iparY,par_stride, par_blk_offset) = inPar(iparY, par_stride, par_blk_offset) + k*(pyin*sina + pxin*(1.-cosa));
    outPar(iparZ,par_stride, par_blk_offset) = zout;

    outPar(iparIpt,par_stride, par_blk_offset)   = inPar(iparIpt, par_stride, par_blk_offset);
    outPar(iparPhi,par_stride, par_blk_offset)   = inPar(iparPhi, par_stride, par_blk_offset) + alpha;
    outPar(iparTheta,par_stride, par_blk_offset) = inPar(iparTheta, par_stride, par_blk_offset);

    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);

    errorProp[PosInMtrx(0,0,6, block_size) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6, block_size) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6, block_size) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6, block_size) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6, block_size) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6, block_size) + it] = 1.0f;

    errorProp[PosInMtrx(0,2,6, block_size) + it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;//2
    errorProp[PosInMtrx(0,3,6, block_size) + it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.-cCosPsina))*(pt*pt);//3
    errorProp[PosInMtrx(0,4,6, block_size) + it] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));//4
    errorProp[PosInMtrx(0,5,6, block_size) + it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)*(icosT*icosT);//5
    errorProp[PosInMtrx(1,2,6, block_size) + it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;//8
    errorProp[PosInMtrx(1,3,6, block_size) + it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.-cCosPsina))*(pt*pt);//9
    errorProp[PosInMtrx(1,4,6, block_size) + it] = (k*pt)*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);//10
    errorProp[PosInMtrx(1,5,6, block_size) + it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);//11
    errorProp[PosInMtrx(4,2,6, block_size) + it] = -inPar(iparIpt, par_stride, par_blk_offset)*sinT*(icosTk);//26
    errorProp[PosInMtrx(4,3,6, block_size) + it] = sinT*deltaZ*(icosTk);//27
    errorProp[PosInMtrx(4,5,6, block_size) + it] = inPar(iparIpt, par_stride, par_blk_offset)*deltaZ*(icosT*icosTk);//29
  }

  MultHelixPropEndcap<MP6x6SFaccessor, block_size>(errorProp, inErr, temp, tid);
  MultHelixPropTranspEndcap<MP6x6SFaccessor, block_size>(errorProp, temp, outErr, tid);

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
                   [=,&trkNacc    = *trkNaccPtr,
                      &hitNacc    = *hitNaccPtr,
                      &outtrkNacc = *outtrkNaccPtr] (const auto i) {
                     for(int layer=0; layer<nlayer; ++layer) {
                       propagateToZ<MPTRKAccessorTp, MPHITAccessorTp, bsize>(outtrkNacc, trkNacc, hitNacc, i, layer);
                       KalmanUpdate<MPTRKAccessorTp, MPHITAccessorTp, bsize>(outtrkNacc, hitNacc, i, layer);
                     }
                   });
#if defined(__NVCOMPILER_CUDA__) 
      convertTracks<order>(outtrk, outtrkNPtr.get());
#endif

   } //end of itr loop

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);


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
