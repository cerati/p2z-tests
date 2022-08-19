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

#include <cuda_runtime.h>
#include <cassert>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <random>


#ifndef ntrks
#define ntrks 9600
#endif

//#define ntrks    (ntrks/bsize)

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

#ifndef num_streams
#define num_streams 1
#endif

#ifndef threadsperblock
#define threadsperblock 32
#endif

namespace impl {

  /**
     Simple array object which mimics std::array
  */
  template <typename T, int n> struct array {
    using value_type = T;
    T data[n];

    constexpr T &operator[](int i) { return data[i]; }
    constexpr const T &operator[](int i) const { return data[i]; }
    constexpr int size() const { return n; }

    array() = default;
    array(const array<T, n> &) = default;
    array(array<T, n> &&) = default;

    array<T, n> &operator=(const array<T, n> &) = default;
    array<T, n> &operator=(array<T, n> &&) = default;
  };
  
  template<typename Tp>
  struct UVMAllocator {
    public:

      typedef Tp value_type;

      UVMAllocator () {};

      UVMAllocator(const UVMAllocator&) { }
       
      template<typename Tp1> constexpr UVMAllocator(const UVMAllocator<Tp1>&) { }

      ~UVMAllocator() { }

      Tp* address(Tp& x) const { return &x; }

      std::size_t  max_size() const throw() { return size_t(-1) / sizeof(Tp); }

      [[nodiscard]] Tp* allocate(std::size_t n){

        Tp* ptr = nullptr;

        auto err = cudaMallocManaged((void **)&ptr,n*sizeof(Tp));

        if( err != cudaSuccess ) {
          ptr = (Tp *) NULL;
          std::cerr << " cudaMallocManaged failed for " << n*sizeof(Tp) << " bytes " <<cudaGetErrorString(err)<< std::endl;
          assert(0);
        }

        return ptr;
      }
      void deallocate( Tp* p, std::size_t n) noexcept {
        cudaFree((void *)p);
        return;
      }
    };
    
   template <typename IntType>
   class counting_iterator {
       static_assert(std::numeric_limits<IntType>::is_integer, "Cannot instantiate counting_iterator with a non-integer type");
     public:
       using value_type = IntType;
       using difference_type = typename std::make_signed<IntType>::type;
       using pointer = IntType*;
       using reference = IntType&;
       using iterator_category = std::random_access_iterator_tag;

       counting_iterator() : value(0) { }
       explicit counting_iterator(IntType v) : value(v) { }

       value_type operator*() const { return value; }
       value_type operator[](difference_type n) const { return value + n; }

       counting_iterator& operator++() { ++value; return *this; }
       counting_iterator operator++(int) {
         counting_iterator result{value};
         ++value;
         return result;
       }  
       counting_iterator& operator--() { --value; return *this; }
       counting_iterator operator--(int) {
         counting_iterator result{value};
         --value;
         return result;
       }
       counting_iterator& operator+=(difference_type n) { value += n; return *this; }
       counting_iterator& operator-=(difference_type n) { value -= n; return *this; }

       friend counting_iterator operator+(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value + n);  }
       friend counting_iterator operator+(difference_type n, counting_iterator const& i)          { return counting_iterator(i.value + n);  }
       friend difference_type   operator-(counting_iterator const& x, counting_iterator const& y) { return x.value - y.value;  }
       friend counting_iterator operator-(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value - n);  }

       friend bool operator==(counting_iterator const& x, counting_iterator const& y) { return x.value == y.value;  }
       friend bool operator!=(counting_iterator const& x, counting_iterator const& y) { return x.value != y.value;  }
       friend bool operator<(counting_iterator const& x, counting_iterator const& y)  { return x.value < y.value; }
       friend bool operator<=(counting_iterator const& x, counting_iterator const& y) { return x.value <= y.value; }
       friend bool operator>(counting_iterator const& x, counting_iterator const& y)  { return x.value > y.value; }
       friend bool operator>=(counting_iterator const& x, counting_iterator const& y) { return x.value >= y.value; }

     private:
       IntType value;
   };

} //impl


enum class FieldOrder{P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER,
                      P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER};
                      
enum class ConversionType{P2Z_CONVERT_TO_INTERNAL_ORDER, P2Z_CONVERT_FROM_INTERNAL_ORDER};   

const std::array<int, 36> SymOffsets66{0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};

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

template <typename T, int N>
struct MPNX_ {
   impl::array<T,N> data;
   //basic accessors
   __device__ __host__ inline const T& operator[](const int idx) const {return data[idx];}
   __device__ __host__ inline T& operator[](const int idx) {return data[idx];}
};

using MP1I_    = MPNX_<int,   1 >;
using MP1F_    = MPNX_<float, 1 >;
using MP2F_    = MPNX_<float, 2 >;
using MP3F_    = MPNX_<float, 3 >;
using MP6F_    = MPNX_<float, 6 >;
using MP2x2SF_ = MPNX_<float, 3 >;
using MP3x3SF_ = MPNX_<float, 6 >;
using MP6x6SF_ = MPNX_<float, 21>;
using MP6x6F_  = MPNX_<float, 36>;
using MP3x3_   = MPNX_<float, 9 >;
using MP3x6_   = MPNX_<float, 18>;

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

using IntAllocator   = impl::UVMAllocator<int>;
using FloatAllocator = impl::UVMAllocator<float>;
using MPTRKAllocator = impl::UVMAllocator<MPTRK_>;
using MPHITAllocator = impl::UVMAllocator<MPHIT_>;

template <typename T, typename Allocator, int n>
struct MPNX {
   using DataType = T;

   static constexpr int N    = n;

   const int nTrks;//note that bSize is a tuning parameter!
   const int nEvts;
   const int nLayers;

   std::vector<T, Allocator> data;

   MPNX() : nTrks(0), nEvts(0), nLayers(0), data(n){}

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

using MP1I    = MPNX<int,  IntAllocator,   1 >;
using MP1F    = MPNX<float,FloatAllocator, 1 >;
using MP2F    = MPNX<float,FloatAllocator, 2 >;
using MP3F    = MPNX<float,FloatAllocator, 3 >;
using MP6F    = MPNX<float,FloatAllocator, 6 >;
using MP3x3   = MPNX<float,FloatAllocator, 9 >;
using MP3x6   = MPNX<float,FloatAllocator, 18>;
using MP2x2SF = MPNX<float,FloatAllocator, 3 >;
using MP3x3SF = MPNX<float,FloatAllocator, 6 >;
using MP6x6SF = MPNX<float,FloatAllocator, 21>;
using MP6x6F  = MPNX<float,FloatAllocator, 36>;


template <typename MPNTp, FieldOrder Order>
struct MPNXAccessor {
   typedef typename MPNTp::DataType T;

   static constexpr int n   = MPNTp::N;//matrix linear dim (total number of els)

   int nTrks;
   int nEvts;
   int nLayers;

   int NevtsNtrks;

   int stride;
   
   int thread_stride;

   T* data_; //accessor field only for the data access, not allocated here

   MPNXAccessor() = default;

   MPNXAccessor(const MPNTp &v) :
        nTrks(v.nTrks),
        nEvts(v.nEvts),
        nLayers(v.nLayers),
        NevtsNtrks(nEvts*nTrks),
        stride(Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER ? nTrks*nEvts*nLayers  : nTrks*nEvts*n),
        thread_stride(Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER ? stride  : NevtsNtrks),              
        data_(const_cast<T*>(v.data.data())){ }

   __device__ __host__ inline T& operator[](const int idx) const {return data_[idx];}

   __device__ __host__ inline T& operator()(const int mat_idx, const int trkev_idx, const int layer_idx) const {
     if      constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return data_[mat_idx*stride + layer_idx*NevtsNtrks + trkev_idx];//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else //(Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return data_[layer_idx*stride + mat_idx*NevtsNtrks + trkev_idx];
   }//i is the internal dof index

   __device__ __host__ inline T& operator()(const int thrd_idx, const int blk_offset) const { return data_[thrd_idx*thread_stride + blk_offset];}//

   __device__ __host__ inline int GetThreadOffset(const int thrd_idx, const int layer_idx = 0) const {
     if      constexpr (Order == FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return (layer_idx*NevtsNtrks + thrd_idx);//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else //(Order == FieldOrder::P2Z_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return (layer_idx*stride + thrd_idx);
   }
   
   __device__ __host__ inline void load(MPNX_<T, n>& dest, const int tid, const int layer = 0) const {
      auto tid_offset = GetThreadOffset(tid, layer);
#pragma unroll
      for(int id = 0; id < n; id++){
          dest[id] = this->operator()(id, tid_offset);
      }
      return;
   }
   __device__ __host__ inline void save(const MPNX_<T, n>& src, const int tid, const int layer = 0){
      auto tid_offset = GetThreadOffset(tid, layer); 
#pragma unroll
      for(int id = 0; id < n; id++){
        this->operator()(id, tid_offset) = src[id];
      }
      return;
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
  
   __device__ __host__ inline const auto load(const int tid) const {
    MPTRK_ dst;

    this->par.load(dst.par, tid, 0);
    this->cov.load(dst.cov, tid, 0);
    this->q.load(dst.q, tid, 0);
    
    return dst;
  }
  
  __device__ __host__ inline void save(MPTRK_ &src, const int tid, const int layer = 0) {
    this->par.save(src.par, tid, layer);
    this->cov.save(src.cov, tid, layer);
    this->q.save(src.q, tid, layer);
    
    return;
  }
};

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
  
  __device__ __host__ inline const auto load(const int tid, const int layer = 0) const {
    MPHIT_ dst;

    this->pos.load(dst.pos, tid, layer);
    this->cov.load(dst.cov, tid, layer);
    
    return dst;
  } 
};


template<FieldOrder order, ConversionType convers_tp>
void convertTracks(std::vector<MPTRK_> &external_order_data, MPTRK* internal_order_data) {
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> ind(new MPTRKAccessor<order>(*internal_order_data));
  // store in element order for bunches of bsize matrices (a la matriplex)
  const int outer_loop_range = nevts*ntrks;
  //
  std::for_each(impl::counting_iterator(0),
                impl::counting_iterator(outer_loop_range),
                [=, exd_ = external_order_data.data(), &ind_ = *ind] (const auto tid) {
                  {
                  //const int l = it+ib*bsize+ie*ntrks*bsize;
                    //par
    	            for (int ip=0;ip<6;++ip) {
    	              if constexpr (convers_tp == ConversionType::P2Z_CONVERT_FROM_INTERNAL_ORDER)
    	                exd_[tid].par.data[ip] = ind_.par(ip, tid, 0);
    	              else
    	                ind_.par(ip, tid, 0) = exd_[tid].par.data[ip];  
    	            }
    	            //cov
    	            for (int ip=0;ip<21;++ip) {
    	              if constexpr (convers_tp == ConversionType::P2Z_CONVERT_FROM_INTERNAL_ORDER)
    	                exd_[tid].cov.data[ip] = ind_.cov(ip, tid, 0);
    	              else
    	                ind_.cov(ip, tid, 0) = exd_[tid].cov.data[ip];
    	            }
    	            //q
    	            if constexpr (convers_tp == ConversionType::P2Z_CONVERT_FROM_INTERNAL_ORDER)
    	              exd_[tid].q.data[0] = ind_.q(0, tid, 0);//fixme check
    	            else
    	              ind_.q(0, tid, 0) = exd_[tid].q.data[0];
                  }
                });
   //
   return;
}


template<FieldOrder order, ConversionType convers_tp>
void convertHits(std::vector<MPHIT_> &external_order_data, MPHIT* internal_order_data) {
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> ind(new MPHITAccessor<order>(*internal_order_data));
  // store in element order for bunches of bsize matrices (a la matriplex)
  const int outer_loop_range = nevts*ntrks;
  
  std::for_each(impl::counting_iterator(0),
                impl::counting_iterator(outer_loop_range),
                [=, exd_ = external_order_data.data(), &ind_ = *ind] (const auto tid) {
                   //  
                   for(int layer=0; layer<nlayer; ++layer) {  
                     {
                       //pos
                       for (int ip=0;ip<3;++ip) {
                         if constexpr (convers_tp == ConversionType::P2Z_CONVERT_FROM_INTERNAL_ORDER)
                           exd_[layer+nlayer*tid].pos.data[ip] = ind_.pos(ip, tid, layer);
                         else
                           ind_.pos(ip, tid, layer) = exd_[layer+nlayer*tid].pos.data[ip];
                       }
                       //cov
                       for (int ip=0;ip<6;++ip) {
                         if constexpr (convers_tp == ConversionType::P2Z_CONVERT_FROM_INTERNAL_ORDER)
                           exd_[layer+nlayer*tid].cov.data[ip] = ind_.cov(ip, tid, layer);
                         else
                           ind_.cov(ip, tid, layer) = exd_[layer+nlayer*tid].cov.data[ip];
                       }
                     } 
                  }
               });
  
  return;
}

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


void prepareTracks(std::vector<MPTRK_> &trcks, ATRK &inputtrk) {
  //
  for (int ie=0;ie<nevts;++ie) {
    for (int ib=0;ib<ntrks;++ib) {
      {
	      //par
	      for (int ip=0;ip<6;++ip) {
	        trcks[ib + ntrks*ie].par.data[ip] = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov, scale by factor 100
	      for (int ip=0;ip<21;++ip) {
	        trcks[ib + ntrks*ie].cov.data[ip] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
	      }
	      //q
	      trcks[ib + ntrks*ie].q.data[0] = inputtrk.q;//can't really smear this or fit will be wrong
      }
    }
  }
  //
  return;
}

void prepareHits(std::vector<MPHIT_> &hits, std::vector<AHIT>& inputhits) {
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (int lay=0;lay<nlayer;++lay) {

    int mylay = lay;
    if (lay>=inputhits.size()) {
      // int wraplay = inputhits.size()/lay;
      exit(1);
    }
    AHIT& inputhit = inputhits[mylay];

    for (int ie=0;ie<nevts;++ie) {
      for (int ib=0;ib<ntrks;++ib) {
        {
        	//pos
        	for (int ip=0;ip<3;++ip) {
        	  hits[lay+nlayer*(ib + ntrks*ie)].pos.data[ip] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (int ip=0;ip<6;++ip) {
        	  hits[lay+nlayer*(ib + ntrks*ie)].cov.data[ip] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return;
}


//////////////////////////////////////////////////////////////////////////////////////
// Aux utils 
MPTRK_* bTk(MPTRK_* tracks, int ev, int ib) {
  return &(tracks[ib + ntrks*ev]);
}

const MPTRK_* bTk(const MPTRK_* tracks, int ev, int ib) {
  return &(tracks[ib + ntrks*ev]);
}

float q(const MP1I_* bq, int it){
  return (*bq).data[0];
}
//
float par(const MP6F_* bpars, int it, int ipar){
  return (*bpars).data[it + ipar];
}
float x    (const MP6F_* bpars, int it){ return par(bpars, it, 0); }
float y    (const MP6F_* bpars, int it){ return par(bpars, it, 1); }
float z    (const MP6F_* bpars, int it){ return par(bpars, it, 2); }
float ipt  (const MP6F_* bpars, int it){ return par(bpars, it, 3); }
float phi  (const MP6F_* bpars, int it){ return par(bpars, it, 4); }
float theta(const MP6F_* bpars, int it){ return par(bpars, it, 5); }
//
float par(const MPTRK_* btracks, int it, int ipar){
  return par(&(*btracks).par,it,ipar);
}
float x    (const MPTRK_* btracks, int it){ return par(btracks, it, 0); }
float y    (const MPTRK_* btracks, int it){ return par(btracks, it, 1); }
float z    (const MPTRK_* btracks, int it){ return par(btracks, it, 2); }
float ipt  (const MPTRK_* btracks, int it){ return par(btracks, it, 3); }
float phi  (const MPTRK_* btracks, int it){ return par(btracks, it, 4); }
float theta(const MPTRK_* btracks, int it){ return par(btracks, it, 5); }
//
float par(const MPTRK_* tracks, int ev, int tk, int ipar){
  int ib = tk;
  const MPTRK_* btracks = bTk(tracks, ev, ib);
  int it = 0;
  return par(btracks, it, ipar);
}
float x    (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 0); }
float y    (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 1); }
float z    (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 2); }
float ipt  (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 3); }
float phi  (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 4); }
float theta(const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 5); }
//

const MPHIT_* bHit(const MPHIT_* hits, int ev, int ib) {
  return &(hits[ib + ntrks*ev]);
}
const MPHIT_* bHit(const MPHIT_* hits, int ev, int ib,int lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*ntrks)]);
}
//
float Pos(const MP3F_* hpos, int it, int ipar){
  return (*hpos).data[it + ipar];
}
float x(const MP3F_* hpos, int it)    { return Pos(hpos, it, 0); }
float y(const MP3F_* hpos, int it)    { return Pos(hpos, it, 1); }
float z(const MP3F_* hpos, int it)    { return Pos(hpos, it, 2); }
//
float Pos(const MPHIT_* hits, int it, int ipar){
  return Pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT_* hits, int it)    { return Pos(hits, it, 0); }
float y(const MPHIT_* hits, int it)    { return Pos(hits, it, 1); }
float z(const MPHIT_* hits, int it)    { return Pos(hits, it, 2); }
//
float Pos(const MPHIT_* hits, int ev, int tk, int ipar){
  int ib = tk;
  const MPHIT_* bhits = bHit(hits, ev, ib);
  int it = 0;
  return Pos(bhits,it,ipar);
}
float x(const MPHIT_* hits, int ev, int tk)    { return Pos(hits, ev, tk, 0); }
float y(const MPHIT_* hits, int ev, int tk)    { return Pos(hits, ev, tk, 1); }
float z(const MPHIT_* hits, int ev, int tk)    { return Pos(hits, ev, tk, 2); }


////////////////////////////////////////////////////////////////////////
///MAIN compute kernels

__device__ inline void MultHelixPropEndcap(const MP6x6F_ &a, const MP6x6SF_ &b, MP6x6F_ &c) {
  {
    c[ 0] = b[ 0] + a[ 2]*b[ 3] + a[ 3]*b[ 6] + a[ 4]*b[10] + a[ 5]*b[15];
    c[ 1] = b[ 1] + a[ 2]*b[ 4] + a[ 3]*b[ 7] + a[ 4]*b[11] + a[ 5]*b[16];
    c[ 2] = b[ 3] + a[ 2]*b[ 5] + a[ 3]*b[ 8] + a[ 4]*b[12] + a[ 5]*b[17];
    c[ 3] = b[ 6] + a[ 2]*b[ 8] + a[ 3]*b[ 9] + a[ 4]*b[13] + a[ 5]*b[18];
    c[ 4] = b[10] + a[ 2]*b[12] + a[ 3]*b[13] + a[ 4]*b[14] + a[ 5]*b[19];
    c[ 5] = b[15] + a[ 2]*b[17] + a[ 3]*b[18] + a[ 4]*b[19] + a[ 5]*b[20];
    c[ 6] = b[ 1] + a[ 8]*b[ 3] + a[ 9]*b[ 6] + a[10]*b[10] + a[11]*b[15];
    c[ 7] = b[ 2] + a[ 8]*b[ 4] + a[ 9]*b[ 7] + a[10]*b[11] + a[11]*b[16];
    c[ 8] = b[ 4] + a[ 8]*b[ 5] + a[ 9]*b[ 8] + a[10]*b[12] + a[11]*b[17];
    c[ 9] = b[ 7] + a[ 8]*b[ 8] + a[ 9]*b[ 9] + a[10]*b[13] + a[11]*b[18];
    c[10] = b[11] + a[ 8]*b[12] + a[ 9]*b[13] + a[10]*b[14] + a[11]*b[19];
    c[11] = b[16] + a[ 8]*b[17] + a[ 9]*b[18] + a[10]*b[19] + a[11]*b[20];
    c[12] = 0.f;
    c[13] = 0.f;
    c[14] = 0.f;
    c[15] = 0.f;
    c[16] = 0.f;
    c[17] = 0.f;
    c[18] = b[ 6];
    c[19] = b[ 7];
    c[20] = b[ 8];
    c[21] = b[ 9];
    c[22] = b[13];
    c[23] = b[18];
    c[24] = a[26]*b[ 3] + a[27]*b[ 6] + b[10] + a[29]*b[15];
    c[25] = a[26]*b[ 4] + a[27]*b[ 7] + b[11] + a[29]*b[16];
    c[26] = a[26]*b[ 5] + a[27]*b[ 8] + b[12] + a[29]*b[17];
    c[27] = a[26]*b[ 8] + a[27]*b[ 9] + b[13] + a[29]*b[18];
    c[28] = a[26]*b[12] + a[27]*b[13] + b[14] + a[29]*b[19];
    c[29] = a[26]*b[17] + a[27]*b[18] + b[19] + a[29]*b[20];
    c[30] = b[15];
    c[31] = b[16];
    c[32] = b[17];
    c[33] = b[18];
    c[34] = b[19];
    c[35] = b[20];
  }
  return;
}

__device__ inline void MultHelixPropTranspEndcap(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SF_ &c) {
  {
    c[ 0] = b[ 0] + b[ 2]*a[ 2] + b[ 3]*a[ 3] + b[ 4]*a[ 4] + b[ 5]*a[ 5];
    c[ 1] = b[ 6] + b[ 8]*a[ 2] + b[ 9]*a[ 3] + b[10]*a[ 4] + b[11]*a[ 5];
    c[ 2] = b[ 7] + b[ 8]*a[ 8] + b[ 9]*a[ 9] + b[10]*a[10] + b[11]*a[11];
    c[ 3] = b[12] + b[14]*a[ 2] + b[15]*a[ 3] + b[16]*a[ 4] + b[17]*a[ 5];
    c[ 4] = b[13] + b[14]*a[ 8] + b[15]*a[ 9] + b[16]*a[10] + b[17]*a[11];
    c[ 5] = 0.f;
    c[ 6] = b[18] + b[20]*a[ 2] + b[21]*a[ 3] + b[22]*a[ 4] + b[23]*a[ 5];
    c[ 7] = b[19] + b[20]*a[ 8] + b[21]*a[ 9] + b[22]*a[10] + b[23]*a[11];
    c[ 8] = 0.f;
    c[ 9] = b[21];
    c[10] = b[24] + b[26]*a[ 2] + b[27]*a[ 3] + b[28]*a[ 4] + b[29]*a[ 5];
    c[11] = b[25] + b[26]*a[ 8] + b[27]*a[ 9] + b[28]*a[10] + b[29]*a[11];
    c[12] = 0.f;
    c[13] = b[27];
    c[14] = b[26]*a[26] + b[27]*a[27] + b[28] + b[29]*a[29];
    c[15] = b[30] + b[32]*a[ 2] + b[33]*a[ 3] + b[34]*a[ 4] + b[35]*a[ 5];
    c[16] = b[31] + b[32]*a[ 8] + b[33]*a[ 9] + b[34]*a[10] + b[35]*a[11];
    c[17] = 0.f;
    c[18] = b[33];
    c[19] = b[32]*a[26] + b[33]*a[27] + b[34] + b[35]*a[29];
    c[20] = b[35];
  }
  return;
}

__device__ inline void KalmanGainInv(const MP6x6SF_ &a, const MP3x3SF_ &b, MP3x3_ &c) {

  {
    double det =
      ((a[0]+b[0])*(((a[ 6]+b[ 3]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[7]+b[4])))) -
      ((a[1]+b[1])*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[2]+b[2])))) +
      ((a[2]+b[2])*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[6]+b[3]))));
    double invdet = 1.0/det;

    c[ 0] =   invdet*(((a[ 6]+b[ 3]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[7]+b[4])));
    c[ 1] =  -invdet*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[2]+b[2]) *(a[7]+b[4])));
    c[ 2] =   invdet*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[7]+b[4])));
    c[ 3] =  -invdet*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[2]+b[2])));
    c[ 4] =   invdet*(((a[ 0]+b[ 0]) *(a[11]+b[5])) - ((a[2]+b[2]) *(a[2]+b[2])));
    c[ 5] =  -invdet*(((a[ 0]+b[ 0]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[1]+b[1])));
    c[ 6] =   invdet*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[6]+b[3])));
    c[ 7] =  -invdet*(((a[ 0]+b[ 0]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[1]+b[1])));
    c[ 8] =   invdet*(((a[ 0]+b[ 0]) *(a[6]+b[3])) - ((a[1]+b[1]) *(a[1]+b[1])));
  }
  
  return;
}

__device__ inline void KalmanGain(const MP6x6SF_ &a, const MP3x3_ &b, MP3x6_ &c) {

  {
    c[ 0] = a[0]*b[0] + a[ 1]*b[3] + a[2]*b[6];
    c[ 1] = a[0]*b[1] + a[ 1]*b[4] + a[2]*b[7];
    c[ 2] = a[0]*b[2] + a[ 1]*b[5] + a[2]*b[8];
    c[ 3] = a[1]*b[0] + a[ 6]*b[3] + a[7]*b[6];
    c[ 4] = a[1]*b[1] + a[ 6]*b[4] + a[7]*b[7];
    c[ 5] = a[1]*b[2] + a[ 6]*b[5] + a[7]*b[8];
    c[ 6] = a[2]*b[0] + a[ 7]*b[3] + a[11]*b[6];
    c[ 7] = a[2]*b[1] + a[ 7]*b[4] + a[11]*b[7];
    c[ 8] = a[2]*b[2] + a[ 7]*b[5] + a[11]*b[8];
    c[ 9] = a[3]*b[0] + a[ 8]*b[3] + a[12]*b[6];
    c[10] = a[3]*b[1] + a[ 8]*b[4] + a[12]*b[7];
    c[11] = a[3]*b[2] + a[ 8]*b[5] + a[12]*b[8];
    c[12] = a[4]*b[0] + a[ 9]*b[3] + a[13]*b[6];
    c[13] = a[4]*b[1] + a[ 9]*b[4] + a[13]*b[7];
    c[14] = a[4]*b[2] + a[ 9]*b[5] + a[13]*b[8];
    c[15] = a[5]*b[0] + a[10]*b[3] + a[14]*b[6];
    c[16] = a[5]*b[1] + a[10]*b[4] + a[14]*b[7];
    c[17] = a[5]*b[2] + a[10]*b[5] + a[14]*b[8];
  }
  
  return;
}

__device__ void KalmanUpdate(MP6x6SF_ &trkErr, MP6F_ &inPar, const MP3x3SF_ &hitErr, const MP3F_ &msP){

  MP3x3_ inverse_temp;
  MP3x6_ kGain;
  MP6x6SF_ newErr;
  
  KalmanGainInv(trkErr, hitErr, inverse_temp);
  KalmanGain(trkErr, inverse_temp, kGain);

  {
    const auto xin     = inPar[iparX];
    const auto yin     = inPar[iparY];
    const auto zin     = inPar[iparZ];
    const auto ptin    = 1.f/ inPar[iparIpt];
    const auto phiin   = inPar[iparPhi];
    const auto thetain = inPar[iparTheta];
    const auto xout    = msP[iparX];
    const auto yout    = msP[iparY];
    //const auto zout    = msP[iparZ];

    auto xnew     = xin + (kGain[0]*(xout-xin)) +(kGain[1]*(yout-yin)); 
    auto ynew     = yin + (kGain[3]*(xout-xin)) +(kGain[4]*(yout-yin)); 
    auto znew     = zin + (kGain[6]*(xout-xin)) +(kGain[7]*(yout-yin)); 
    auto ptnew    = ptin + (kGain[9]*(xout-xin)) +(kGain[10]*(yout-yin)); 
    auto phinew   = phiin + (kGain[12]*(xout-xin)) +(kGain[13]*(yout-yin)); 
    auto thetanew = thetain + (kGain[15]*(xout-xin)) +(kGain[16]*(yout-yin)); 

    newErr[ 0] = trkErr[ 0] - (kGain[ 0]*trkErr[0]+kGain[1]*trkErr[1]+kGain[2]*trkErr[2]);
    newErr[ 1] = trkErr[ 1] - (kGain[ 0]*trkErr[1]+kGain[1]*trkErr[6]+kGain[2]*trkErr[7]);
    newErr[ 2] = trkErr[ 2] - (kGain[ 0]*trkErr[2]+kGain[1]*trkErr[7]+kGain[2]*trkErr[11]);
    newErr[ 3] = trkErr[ 3] - (kGain[ 0]*trkErr[3]+kGain[1]*trkErr[8]+kGain[2]*trkErr[12]);
    newErr[ 4] = trkErr[ 4] - (kGain[ 0]*trkErr[4]+kGain[1]*trkErr[9]+kGain[2]*trkErr[13]);
    newErr[ 5] = trkErr[ 5] - (kGain[ 0]*trkErr[5]+kGain[1]*trkErr[10]+kGain[2]*trkErr[14]);

    newErr[ 6] = trkErr[ 6] - (kGain[ 3]*trkErr[1]+kGain[4]*trkErr[6]+kGain[5]*trkErr[7]);
    newErr[ 7] = trkErr[ 7] - (kGain[ 3]*trkErr[2]+kGain[4]*trkErr[7]+kGain[5]*trkErr[11]);
    newErr[ 8] = trkErr[ 8] - (kGain[ 3]*trkErr[3]+kGain[4]*trkErr[8]+kGain[5]*trkErr[12]);
    newErr[ 9] = trkErr[ 9] - (kGain[ 3]*trkErr[4]+kGain[4]*trkErr[9]+kGain[5]*trkErr[13]);
    newErr[10] = trkErr[10] - (kGain[ 3]*trkErr[5]+kGain[4]*trkErr[10]+kGain[5]*trkErr[14]);

    newErr[11] = trkErr[11] - (kGain[ 6]*trkErr[2]+kGain[7]*trkErr[7]+kGain[8]*trkErr[11]);
    newErr[12] = trkErr[12] - (kGain[ 6]*trkErr[3]+kGain[7]*trkErr[8]+kGain[8]*trkErr[12]);
    newErr[13] = trkErr[13] - (kGain[ 6]*trkErr[4]+kGain[7]*trkErr[9]+kGain[8]*trkErr[13]);
    newErr[14] = trkErr[14] - (kGain[ 6]*trkErr[5]+kGain[7]*trkErr[10]+kGain[8]*trkErr[14]);

    newErr[15] = trkErr[15] - (kGain[ 9]*trkErr[3]+kGain[10]*trkErr[8]+kGain[11]*trkErr[12]);
    newErr[16] = trkErr[16] - (kGain[ 9]*trkErr[4]+kGain[10]*trkErr[9]+kGain[11]*trkErr[13]);
    newErr[17] = trkErr[17] - (kGain[ 9]*trkErr[5]+kGain[10]*trkErr[10]+kGain[11]*trkErr[14]);

    newErr[18] = trkErr[18] - (kGain[12]*trkErr[4]+kGain[13]*trkErr[9]+kGain[14]*trkErr[13]);
    newErr[19] = trkErr[19] - (kGain[12]*trkErr[5]+kGain[13]*trkErr[10]+kGain[14]*trkErr[14]);

    newErr[20] = trkErr[20] - (kGain[15]*trkErr[5]+kGain[16]*trkErr[10]+kGain[17]*trkErr[14]);
    
    inPar[iparX]     = xnew;
    inPar[iparY]     = ynew;
    inPar[iparZ]     = znew;
    inPar[iparIpt]   = ptnew;
    inPar[iparPhi]   = phinew;
    inPar[iparTheta] = thetanew;
    
 #pragma unroll
    for (int i = 0; i < 21; i++){
      trkErr[ i] = trkErr[ i] - newErr[ i];
    }

  }
  
  return;
}              

//constexpr auto kfact= 100/(-0.299792458*3.8112);
constexpr auto kfact= 100/3.8;

__device__ void propagateToZ(const MP6x6SF_ &inErr, const MP6F_ &inPar, const MP1I_ &inChg, 
                  const MP3F_ &msP, MP6x6SF_ &outErr, MP6F_ &outPar) {
  
  MP6x6F_ errorProp;
  MP6x6F_ temp;
 
  auto PosInMtrx = [=] (int i, int j, int D) constexpr {return (i*D+j);};
//#pragma omp simd
  {	
    const auto zout = msP[iparZ];
    //note: in principle charge is not needed and could be the sign of ipt
    const auto k = inChg[0]*kfact;
    const auto deltaZ = zout - inPar[iparZ];
    const auto ipt  = inPar[iparIpt];
    const auto pt   = 1.f/ipt;
    const auto phi  = inPar[iparPhi];
    const auto cosP = cosf(phi);
    const auto sinP = sinf(phi);
    const auto theta= inPar[iparTheta];
    const auto cosT = cosf(theta);
    const auto sinT = sinf(theta);
    const auto pxin = cosP*pt;
    const auto pyin = sinP*pt;
    const auto icosT  = 1.f/cosT;
    const auto icosTk = icosT/k;
    const auto alpha  = deltaZ*sinT*ipt*icosTk;
    //const auto alpha = deltaZ*sinT*ipt(inPar]/(cosT*k);
    const auto sina = sinf(alpha); // this can be approximated;
    const auto cosa = cosf(alpha); // this can be approximated;
    //
    outPar[iparX]     = inPar[iparX] + k*(pxin*sina - pyin*(1.f-cosa));
    outPar[iparY]     = inPar[iparY] + k*(pyin*sina + pxin*(1.f-cosa));
    outPar[iparZ]     = zout;
    outPar[iparIpt]   = ipt;
    outPar[iparPhi]   = phi +alpha;
    outPar[iparTheta] = theta;
    
    const auto sCosPsina = sinf(cosP*sina);
    const auto cCosPsina = cosf(cosP*sina);
    
    //for (size_t i=0;i<6;++i) errorProp[bsize*PosInMtrx(i,i,6) + it] = 1.;
    errorProp[PosInMtrx(0,0,6)] = 1.0f;
    errorProp[PosInMtrx(1,1,6)] = 1.0f;
    errorProp[PosInMtrx(2,2,6)] = 1.0f;
    errorProp[PosInMtrx(3,3,6)] = 1.0f;
    errorProp[PosInMtrx(4,4,6)] = 1.0f;
    errorProp[PosInMtrx(5,5,6)] = 1.0f;
    //
    errorProp[PosInMtrx(0,1,6)] = 0.f;
    errorProp[PosInMtrx(0,2,6)] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)*icosT;
    errorProp[PosInMtrx(0,3,6)] = cosP*sinT*deltaZ*cosa*(1.f-sinP*sCosPsina)*(icosT*pt)-k*(cosP*sina-sinP*(1.f-cCosPsina))*(pt*pt);
    errorProp[PosInMtrx(0,4,6)] = (k*pt)*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.f-cCosPsina));
    errorProp[PosInMtrx(0,5,6)] = cosP*deltaZ*cosa*(1.f-sinP*sCosPsina)*(icosT*icosT);
    errorProp[PosInMtrx(1,2,6)] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)*icosT;
    errorProp[PosInMtrx(1,3,6)] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*pt)-k*(sinP*sina+cosP*(1.f-cCosPsina))*(pt*pt);
    errorProp[PosInMtrx(1,4,6)] = (k*pt)*(-sinP*(1.f-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp[PosInMtrx(1,5,6)] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)*(icosT*icosT);
    errorProp[PosInMtrx(4,2,6)] = -ipt*sinT*(icosTk);//!
    errorProp[PosInMtrx(4,3,6)] = sinT*deltaZ*(icosTk);
    errorProp[PosInMtrx(4,5,6)] = ipt*deltaZ*(icosT*icosTk);//!
  }
  
  MultHelixPropEndcap(errorProp, inErr, temp);
  MultHelixPropTranspEndcap(errorProp, temp, outErr);
  
  return;
}


template <int layers, FieldOrder order = FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER, bool grid_stride = true>
__global__ void launch_p2z_kernels(MPTRKAccessor<order> &obtracksAcc, MPTRKAccessor<order> &btracksAcc, MPHITAccessor<order> &bhitsAcc, const int length){
   auto i = threadIdx.x + blockIdx.x * blockDim.x;

   MPTRK_ obtracks;

   while (i < length) {
     //
     const auto& btracks = btracksAcc.load(i);
     
     for(int layer = 0; layer < layers; ++layer) {  
       //
       const auto& bhits = bhitsAcc.load(i, layer);
       //
       propagateToZ(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
       KalmanUpdate(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
       //
     }
     //
     obtracksAcc.save(obtracks, i);
     
     if constexpr (grid_stride) i += gridDim.x * blockDim.x;
     else break;
  }
  return;
}

void p2z_check_error(){
  //	
  auto error = cudaGetLastError();
  if(error != cudaSuccess) std::cout << "Error detected, error " << error << std::endl;
  //
  return;
}


int main (int argc, char* argv[]) {

   #include "input_track.h"

   std::vector<AHIT> inputhits{inputhit21,inputhit20,inputhit19,inputhit18,inputhit17,inputhit16,inputhit15,inputhit14,
                               inputhit13,inputhit12,inputhit11,inputhit10,inputhit09,inputhit08,inputhit07,inputhit06,
                               inputhit05,inputhit04,inputhit03,inputhit02,inputhit01,inputhit00};

   printf("track in pos: x=%f, y=%f, z=%f, r=%f, pt=%f, phi=%f, theta=%f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2],
	  sqrtf(inputtrk.par[0]*inputtrk.par[0] + inputtrk.par[1]*inputtrk.par[1]),
	  1./inputtrk.par[3], inputtrk.par[4], inputtrk.par[5]);
   printf("track in cov: xx=%.2e, yy=%.2e, zz=%.2e \n", inputtrk.cov[SymOffsets66[0]],
	                                       inputtrk.cov[SymOffsets66[(1*6+1)]],
	                                       inputtrk.cov[SymOffsets66[(2*6+2)]]);
   for (int lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;

   constexpr auto order = FieldOrder::P2Z_TRACKBLK_EVENT_LAYER_MATIDX_ORDER;

   using MPTRKAccessorTp = MPTRKAccessor<order>;
   using MPHITAccessorTp = MPHITAccessor<order>;

   impl::UVMAllocator<MPTRKAccessorTp> mptrk_uvm_alloc;
   impl::UVMAllocator<MPHITAccessorTp> mphit_uvm_alloc;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   std::unique_ptr<MPTRK> trcksPtr(new MPTRK(ntrks, nevts));
   auto trcksAccPtr = std::allocate_shared<MPTRKAccessorTp>(mptrk_uvm_alloc, *trcksPtr);
   //
   std::unique_ptr<MPHIT> hitsPtr(new MPHIT(ntrks, nevts, nlayer));
   auto hitsAccPtr = std::allocate_shared<MPHITAccessorTp>(mphit_uvm_alloc, *hitsPtr);
   //
   std::unique_ptr<MPTRK> outtrcksPtr(new MPTRK(ntrks, nevts));
   auto outtrcksAccPtr = std::allocate_shared<MPTRKAccessorTp>(mptrk_uvm_alloc, *outtrcksPtr);
   //
   std::vector<MPTRK_> trcks(nevts*ntrks); 
   prepareTracks(trcks, inputtrk);
   //
   std::vector<MPHIT_> hits(nlayer*nevts*ntrks);
   prepareHits(hits, inputhits);
   //
   std::vector<MPTRK_> outtrcks(nevts*ntrks);
   
   convertHits<order,   ConversionType::P2Z_CONVERT_TO_INTERNAL_ORDER>(hits,     hitsPtr.get());
   convertTracks<order, ConversionType::P2Z_CONVERT_TO_INTERNAL_ORDER>(trcks,    trcksPtr.get());
   convertTracks<order, ConversionType::P2Z_CONVERT_TO_INTERNAL_ORDER>(outtrcks, outtrcksPtr.get());

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*ntrks*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*ntrks*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*ntrks*sizeof(MPHIT));

   const int phys_length      = nevts*ntrks;
   const int outer_loop_range = phys_length;
   //
   dim3 blocks(threadsperblock, 1, 1);
   dim3 grid(((outer_loop_range + threadsperblock - 1)/ threadsperblock),1,1);
   // A warmup run to migrate data on the device
   launch_p2z_kernels<nlayer><<<grid, blocks>>>(*outtrcksAccPtr, *trcksAccPtr, *hitsAccPtr, phys_length);

   cudaDeviceSynchronize();

   p2z_check_error();

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(int itr=0; itr<NITER; itr++) {

     launch_p2z_kernels<nlayer><<<grid, blocks>>>(*outtrcksAccPtr, *trcksAccPtr, *hitsAccPtr, phys_length);

   } //end of itr loop

   cudaDeviceSynchronize();
   p2z_check_error();

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;   

   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, 1, ntrks, wall_time, (setup_stop-setup_start)*0.001, -1);

   convertTracks<order, ConversionType::P2Z_CONVERT_FROM_INTERNAL_ORDER>(outtrcks, outtrcksPtr.get());
   auto outtrk = outtrcks.data();

   int nnans = 0, nfail = 0;
   float avgx = 0, avgy = 0, avgz = 0, avgr = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0, avgdr = 0;

   for (int ie=0;ie<nevts;++ie) {
     for (int it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       float pt_ = std::abs(1./ipt(outtrk,ie, it));
       float phi_ = phi(outtrk,ie,it);
       float theta_ = theta(outtrk,ie,it);
       float hx_ = inputhits[nlayer-1].pos[0];
       float hy_ = inputhits[nlayer-1].pos[1];
       float hz_ = inputhits[nlayer-1].pos[2];
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
 
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
	   fabs( (z_-hz_)/hz_ )>1.) {
	 nfail++;
	 continue;
       }

       avgpt += pt_;
       avgphi += phi_;
       avgtheta += theta_;
       avgx += x_;
       avgy += y_;
       avgz += z_;
       avgr += r_;
       avgdx += (x_-hx_)/x_;
       avgdy += (y_-hy_)/y_;
       avgdz += (z_-hz_)/z_;
       avgdr += (r_-hr_)/r_;
       //if((it+ie*ntrks) < 64) printf("iTrk = %i,  track (x,y,z,r)=(%.6f,%.6f,%.6f,%.6f) \n", it+ie*ntrks, x_,y_,z_,r_);
     }
   }

   avgpt = avgpt/float(nevts*ntrks);
   avgphi = avgphi/float(nevts*ntrks);
   avgtheta = avgtheta/float(nevts*ntrks);
   avgx = avgx/float(nevts*ntrks);
   avgy = avgy/float(nevts*ntrks);
   avgz = avgz/float(nevts*ntrks);
   avgr = avgr/float(nevts*ntrks);
   avgdx = avgdx/float(nevts*ntrks);
   avgdy = avgdy/float(nevts*ntrks);
   avgdz = avgdz/float(nevts*ntrks);
   avgdr = avgdr/float(nevts*ntrks);

   float stdx = 0, stdy = 0, stdz = 0, stdr = 0;
   float stddx = 0, stddy = 0, stddz = 0, stddr = 0;
   for (int ie=0;ie<nevts;++ie) {
     for (int it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       float hx_ = inputhits[nlayer-1].pos[0];
       float hy_ = inputhits[nlayer-1].pos[1];
       float hz_ = inputhits[nlayer-1].pos[2];
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
          std::isfinite(y_)==false ||
          std::isfinite(z_)==false
          ) {
        continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
	   fabs( (y_-hy_)/hy_ )>1. ||
	   fabs( (z_-hz_)/hz_ )>1.) {
	 continue;
       }
       stdx += (x_-avgx)*(x_-avgx);
       stdy += (y_-avgy)*(y_-avgy);
       stdz += (z_-avgz)*(z_-avgz);
       stdr += (r_-avgr)*(r_-avgr);
       stddx += ((x_-hx_)/x_-avgdx)*((x_-hx_)/x_-avgdx);
       stddy += ((y_-hy_)/y_-avgdy)*((y_-hy_)/y_-avgdy);
       stddz += ((z_-hz_)/z_-avgdz)*((z_-hz_)/z_-avgdz);
       stddr += ((r_-hr_)/r_-avgdr)*((r_-hr_)/r_-avgdr);
     }
   }

   stdx = sqrtf(stdx/float(nevts*ntrks));
   stdy = sqrtf(stdy/float(nevts*ntrks));
   stdz = sqrtf(stdz/float(nevts*ntrks));
   stdr = sqrtf(stdr/float(nevts*ntrks));
   stddx = sqrtf(stddx/float(nevts*ntrks));
   stddy = sqrtf(stddy/float(nevts*ntrks));
   stddz = sqrtf(stddz/float(nevts*ntrks));
   stddr = sqrtf(stddr/float(nevts*ntrks));

   printf("track x avg=%f std/avg=%f\n", avgx, fabs(stdx/avgx));
   printf("track y avg=%f std/avg=%f\n", avgy, fabs(stdy/avgy));
   printf("track z avg=%f std/avg=%f\n", avgz, fabs(stdz/avgz));
   printf("track r avg=%f std/avg=%f\n", avgr, fabs(stdr/avgz));
   printf("track dx/x avg=%f std=%f\n", avgdx, stddx);
   printf("track dy/y avg=%f std=%f\n", avgdy, stddy);
   printf("track dz/z avg=%f std=%f\n", avgdz, stddz);
   printf("track dr/r avg=%f std=%f\n", avgdr, stddr);
   printf("track pt avg=%f\n", avgpt);
   printf("track phi avg=%f\n", avgphi);
   printf("track theta avg=%f\n", avgtheta);
   printf("number of tracks with nans=%i\n", nnans);
   printf("number of tracks failed=%i\n", nfail);

   return 0;
}
