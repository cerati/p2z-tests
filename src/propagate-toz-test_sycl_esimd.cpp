/*
export PSTL_USAGE_WARNINGS=1
export ONEDPL_USE_DPCPP_BACKEND=1

clang++ -fsycl -O3 -std=c++17 src/propagate-toz-test_sycl_esimd.cpp -o test-sycl.exe -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=16 -Dnlayer=20

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <concepts>
//#include <ranges>

#include <vector>
#include <memory>
#include <numeric>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;

#ifndef bsize
constexpr int bSize = 16;
#else
constexpr int bSize = bsize;
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

template <bool is_sycl_target>
concept SYCLCompute = is_sycl_target == true;

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

template <typename T, int N, int bSize_>
struct MPNX {
   std::array<T,N*bSize_> data;
   //basic accessors
   const T& operator[](const int idx) const {return data[idx];}
   T& operator[](const int idx) {return data[idx];}
   const T& operator()(const int m, const int b) const {return data[m*bSize+b];}
   T& operator()(const int m, const int b) {return data[m*bSize+b];}
   //   
   [[intel::sycl_explicit_simd]] void simd_load(MPNX<simd<T, bSize_>, N, 1>& dst){
#pragma unroll
     for (int ip=0;ip<N;++ip) {    	
       dst.data[ip] = block_load<T, bSize_>(&data[ip*bSize_]); //this->operator()(ip, 0);  
     }//
     return;
   }
//
   [[intel::sycl_explicit_simd]] void simd_save(const MPNX<simd<T, bSize_>, N, 1>& src) {
#pragma unroll
     for (int ip=0;ip<N;++ip) {    	
       block_store<T, bSize_>(&data[ip*bSize_], src.data[ip]); 
     }
     
     return;
   }

};

using MP1I    = MPNX<int,   1 , bSize>;
using MP1F    = MPNX<float, 1 , bSize>;
using MP2F    = MPNX<float, 2 , bSize>;
using MP3F    = MPNX<float, 3 , bSize>;
using MP6F    = MPNX<float, 6 , bSize>;
using MP2x2SF = MPNX<float, 3 , bSize>;
using MP3x3SF = MPNX<float, 6 , bSize>;
using MP6x6SF = MPNX<float, 21, bSize>;
using MP6x6F  = MPNX<float, 36, bSize>;
using MP3x3   = MPNX<float, 9 , bSize>;
using MP3x6   = MPNX<float, 18, bSize>;

// Native fields:
using MP1I_    = MPNX<simd<int, bSize>,   1 , 1>;
using MP1F_    = MPNX<simd<float, bSize>, 1 , 1>;
using MP2F_    = MPNX<simd<float, bSize>, 2 , 1>;
using MP3F_    = MPNX<simd<float, bSize>, 3 , 1>;
using MP6F_    = MPNX<simd<float, bSize>, 6 , 1>;
using MP2x2SF_ = MPNX<simd<float, bSize>, 3 , 1>;
using MP3x3SF_ = MPNX<simd<float, bSize>, 6 , 1>;
using MP6x6SF_ = MPNX<simd<float, bSize>, 21, 1>;
using MP6x6F_  = MPNX<simd<float, bSize>, 36, 1>;
using MP3x3_   = MPNX<simd<float, bSize>, 9 , 1>;
using MP3x6_   = MPNX<simd<float, bSize>, 18, 1>;

struct MPTRK_ {
  MP6F_    par;
  MP6x6SF_ cov;
  MP1I_    q;
};

struct MPHIT_ {
  MP3F_    pos;
  MP3x3SF_ cov;

};


struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK() = default; 
  //
  const MPTRK_ load(){
  
    MPTRK_ dst;
    //
    par.simd_load(dst.par);
    cov.simd_load(dst.cov);
    q.simd_load(dst.q);
    //   
    return std::move(dst);	  
  }
  //
  void save(const MPTRK_ &src){
    //
    par.simd_save(src.par);
    cov.simd_save(src.cov);
    q.simd_save(src.q);
    //
    return;
  }
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
  //
  //
  MPHIT() = default;
  
  const MPHIT_ load(){
    //
    MPHIT_ dst;
    //
    pos.simd_load(dst.pos);
    cov.simd_load(dst.cov);
    //
    return std::move(dst);
  }
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
    W = std::pow (U1, 2) + std::pow (U2, 2);
  }
  while (W >= 1 || W == 0); 
  mult = std::sqrt ((-2 * std::log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult; 
  call = !call; 
  return (mu + sigma * (float) X1);
}


template<typename MPTRKAllocator>
void prepareTracks(std::vector<MPTRK, MPTRKAllocator> &trcks, ATRK &inputtrk) {
  //
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
	      //par
	      for (size_t ip=0;ip<6;++ip) {
	        trcks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov, scale by factor 100
	      for (size_t ip=0;ip<21;++ip) {
	        trcks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
	      }
	      //q
	      trcks[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//can't really smear this or fit will be wrong
      }
    }
  }
  //
  return;
}

template<typename MPHITAllocator>
void prepareHits(std::vector<MPHIT, MPHITAllocator> &hits, AHIT& inputhit) {
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  hits[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  hits[lay+nlayer*(ib + nb*ie)].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
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

float q(const MP1I* bq, size_t it){
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

[[intel::sycl_explicit_simd]] inline void MultHelixPropEndcap(const MP6x6F_ &a, const MP6x6SF_ &b, MP6x6F_ &c) {
//#pragma omp simd 
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

[[intel::sycl_explicit_simd]] inline void MultHelixPropTranspEndcap(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SF_ &c) {

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
  
  return;
}

template<int N = bSize>
[[intel::sycl_explicit_simd]] inline void KalmanGainInv(const MP6x6SF_ &a, const MP3x3SF_ &b, MP3x3_ &c){

  using FloatN = simd<float,N>;
  
  FloatN det =
      ((a[0]+b[0])*(((a[ 6]+b[ 3]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[7]+b[4])))) -
      ((a[1]+b[1])*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[2]+b[2])))) +
      ((a[2]+b[2])*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[6]+b[3]))));
      
  FloatN invdet = esimd::inv(det);

  c[ 0] =   invdet*(((a[ 6]+b[ 3]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[7]+b[4])));
  c[ 1] =  -invdet*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[2]+b[2]) *(a[7]+b[4])));
  c[ 2] =   invdet*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[7]+b[4])));
  c[ 3] =  -invdet*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[2]+b[2])));
  c[ 4] =   invdet*(((a[ 0]+b[ 0]) *(a[11]+b[5])) - ((a[2]+b[2]) *(a[2]+b[2])));
  c[ 5] =  -invdet*(((a[ 0]+b[ 0]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[1]+b[1])));
  c[ 6] =   invdet*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[6]+b[3])));
  c[ 7] =  -invdet*(((a[ 0]+b[ 0]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[1]+b[1])));
  c[ 8] =   invdet*(((a[ 0]+b[ 0]) *(a[6]+b[3])) - ((a[1]+b[1]) *(a[1]+b[1])));
  
  return;
}

[[intel::sycl_explicit_simd]] inline void KalmanGain(const MP6x6SF_ &a, const MP3x3_ &b, MP3x6_ &c) {

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
  
  return;
}

template<int N = bSize>
[[intel::sycl_explicit_simd]] void KalmanUpdate(MP6x6SF_ &trkErr, MP6F_ &inPar, const MP3x3SF_ &hitErr, const MP3F_ &msP) {

  using FloatN = simd<float,N>;
  
  MP3x3_ inverse_temp;
  MP3x6_ kGain;
  MP6x6SF_ newErr;
  
  KalmanGainInv<N>(trkErr, hitErr, inverse_temp);
  KalmanGain(trkErr, inverse_temp, kGain);

//#pragma omp simd
  {
    const FloatN xin     = inPar[iparX];
    const FloatN yin     = inPar[iparY];
    const FloatN zin     = inPar[iparZ];
    const FloatN ptin    = esimd::inv( inPar[iparIpt]);
    const FloatN phiin   = inPar[iparPhi];
    const FloatN thetain = inPar[iparTheta];
    const FloatN xout    = msP[iparX];
    const FloatN yout    = msP[iparY];
    //const FloatN zout    = msP[iparZ];

    FloatN xnew     = xin     + (kGain[ 0]*(xout-xin)) +(kGain[ 1]*(yout-yin)); 
    FloatN ynew     = yin     + (kGain[ 3]*(xout-xin)) +(kGain[ 4]*(yout-yin)); 
    FloatN znew     = zin     + (kGain[ 6]*(xout-xin)) +(kGain[ 7]*(yout-yin)); 
    FloatN ptnew    = ptin    + (kGain[ 9]*(xout-xin)) +(kGain[10]*(yout-yin)); 
    FloatN phinew   = phiin   + (kGain[12]*(xout-xin)) +(kGain[13]*(yout-yin)); 
    FloatN thetanew = thetain + (kGain[15]*(xout-xin)) +(kGain[16]*(yout-yin)); 

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
      trkErr[i] = trkErr[i] - newErr[i];
    }

  }
  
  return;
}              

constexpr float kfact= 100/3.8f;

template<int N = bSize>
[[intel::sycl_explicit_simd]] void propagateToZ(const MP6x6SF_ &inErr, const MP6F_ &inPar, const MP1I_ &inChg, 
                  const MP3F_ &msP, MP6x6SF_ &outErr, MP6F_ &outPar) {
                  
  using FloatN = simd<float,N>;
  
  MP6x6F_ errorProp;
  MP6x6F_ temp;
  
  auto PosInMtrx = [=] (int i, int j, int D) constexpr {return (i*D+j);};
//#pragma omp simd
  {	
    const FloatN zout = msP[iparZ];
    //note: in principle charge is not needed and could be the sign of ipt
    const FloatN k = inChg[0]*kfact;
    const FloatN deltaZ = zout - inPar[iparZ];
    const FloatN ipt  = inPar[iparIpt];
    const FloatN pt   = esimd::inv(ipt);
    const FloatN phi  = inPar[iparPhi];
    const FloatN cosP = esimd::cos(phi);
    const FloatN sinP = esimd::sin(phi);
    const FloatN theta= inPar[iparTheta];
    const FloatN cosT = esimd::cos(theta);
    const FloatN sinT = esimd::sin(theta);
    const FloatN pxin = cosP*pt;
    const FloatN pyin = sinP*pt;
    const FloatN icosT  = esimd::inv(cosT);
    const FloatN icosTk = icosT / k;
    const FloatN alpha  = deltaZ*sinT*ipt*icosTk;
    //const FloatN alpha = deltaZ*sinT*ipt(inPar,it)/(cosT*k);
    const FloatN sina = esimd::sin(alpha); // this can be approximated;
    const FloatN cosa = esimd::cos(alpha); // this can be approximated;
    //
    outPar[iparX]     = inPar[iparX] + k*(pxin*sina - pyin*(1.f-cosa));
    outPar[iparY]     = inPar[iparY] + k*(pyin*sina + pxin*(1.f-cosa));
    outPar[iparZ]     = zout;
    outPar[iparIpt]   = ipt;
    outPar[iparPhi]   = phi +alpha;
    outPar[iparTheta] = theta;
    
    const FloatN sCosPsina = esimd::sin(cosP*sina);
    const FloatN cCosPsina = esimd::cos(cosP*sina);
    
    //for (int i=0;i<6;++i) errorProp[bsize*PosInMtrx(i,i,6) + it] = 1.;
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

class ESIMDSelector : public device_selector {
  // Require GPU device unless HOST is requested in SYCL_DEVICE_FILTER env
  virtual int operator()(const sycl::device &device) const {
    if (const char *dev_filter = getenv("SYCL_DEVICE_FILTER")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      if (filter_string.find("host") != std::string::npos)
        return device.is_host() ? 1000 : -1;
      std::cerr
          << "Supported 'SYCL_DEVICE_FILTER' env var values are 'gpu' and "
             "'host', '"
          << filter_string << "' does not contain such substrings.\n";
      return -1;
    }
    // If "SYCL_DEVICE_FILTER" not defined, only allow gpu device
    return device.is_gpu() ? 1000 : -1;
  }
};

auto exception_handler = [](exception_list l) {
  for (auto ep : l) {
    try {
      std::rethrow_exception(ep);
    } catch (sycl::exception &e0) {
      std::cout << "sycl::exception: " << e0.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "std::exception: " << e.what() << std::endl;
    } catch (...) {
      std::cout << "generic exception\n";
    }
  }
};


int main (int argc, char* argv[]) {

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
   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66[(0)]],
                                              inputtrk.cov[SymOffsets66[(1*6+1)]],
                                              inputtrk.cov[SymOffsets66[(2*6+2)]]);
   printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);
      
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;
   //
   sycl::queue cq(ESIMDSelector{}, exception_handler); //(sycl::gpu_selector{});
   //
   sycl::usm_allocator<MPTRK, sycl::usm::alloc::shared, 16u> MPTRKAllocator(cq);
   sycl::usm_allocator<MPHIT, sycl::usm::alloc::shared, 16u> MPHITAllocator(cq);
   //
   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   //
   std::vector<MPTRK, decltype(MPTRKAllocator)> trcks(nevts*nb, MPTRKAllocator); 
   prepareTracks<decltype(MPTRKAllocator)>(trcks, inputtrk);
   //
   std::vector<MPHIT, decltype(MPHITAllocator)> hits(nlayer*nevts*nb, MPHITAllocator);
   prepareHits<decltype(MPHITAllocator)>(hits, inputhit);
   //
   std::vector<MPTRK, decltype(MPTRKAllocator)> outtrcks(nevts*nb, MPTRKAllocator);
 
   constexpr unsigned outer_loop_range = nevts*ntrks;
   //  
   constexpr unsigned GroupSize = 4;
   // We need that many task groups
   sycl::range<1> GroupRange{outer_loop_range / bSize};
   // We need that many tasks in each group
   sycl::range<1> TaskRange{GroupSize};
   //
   sycl::nd_range<1> Range{GroupRange, TaskRange};
 
   auto p2z_kernels = [=,btracksPtr    = trcks.data(),
                         outtracksPtr  = outtrcks.data(),
                         bhitsPtr      = hits.data()] (const nd_item<1> ndi) [[intel::sycl_explicit_simd]] {
                         //  
                         const int i = ndi.get_global_id(0);

                         MPTRK_ obtracks;
                         //
                         const MPTRK_ btracks = btracksPtr[i].load();
                         //
                         constexpr int N = bsize;
                         //
                         for(int layer=0; layer<nlayer; ++layer) {
                           //
                           const MPHIT_ bhits = bhitsPtr[layer+nlayer*i].load();
                           //
                           propagateToZ<bSize>(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
                           KalmanUpdate<bSize>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                           //
                         }
                         //
                         outtracksPtr[i].save(obtracks);
                       };

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   // A warmup run to migrate data on the device:
   cq.submit([&](sycl::handler &h){
       h.parallel_for(Range, p2z_kernels);
     });
  
   cq.wait();

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(int itr=0; itr<NITER; itr++) {
     cq.submit([&](sycl::handler &h){
       h.parallel_for(Range, p2z_kernels);
     });
   } //end of itr loop

   cq.wait();
   
   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;   

   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);

   auto outtrk = outtrcks.data();
   auto hit    = hits.data();   
   
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

   return 0;
}
