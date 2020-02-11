/*
icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#if USE_GPU
#include "propagateGPU.h"
#include <cuda_profiler_api.h>
#include "cuda_runtime.h"
//#include "propagateGPU.cu"
#else
#include "propagate-toz-test.h"
#endif

int main (int argc, char* argv[]) {

   //ATRK inputtrk;// = new ATRK;
   //AHIT inputhit;// = new AHIT;
   //cudaMallocManaged((void**)&inputtrk,sizeof(ATRK));
   //cudaMallocManaged((void**)&inputhit,sizeof(AHIT));
   ATRK inputtrk = {
     {-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975},
     {6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
      6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,
      0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348},
     1,
     {1, 0, 17, 16, 36, 35, 33, 34, 59, 58, 70, 85, 101, 102, 116, 117, 132, 133, 152, 169, 187, 202}
   };

   AHIT inputhit = {
     {-20.7824649810791, -12.24150276184082, 57.8067626953125},
     {2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945}
   };
//    auto trk_par = std::initializer_list<float>({-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975});
//    
//    auto trk_cov = std::initializer_list<float>({6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
//                   6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,
//                    0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348});
//    inputtrk.q = 1;
//    auto trk_hitidx = std::initializer_list<float>({1, 0, 17, 16, 36, 35, 33, 34, 59, 58, 70, 85, 101, 102, 116, 117, 132, 133, 152, 169, 187, 202});
//
//    std::copy(trk_par.begin(), trk_par.end(),inputtrk.par);
//    std::copy(trk_cov.begin(), trk_cov.end(),inputtrk.cov);
//    std::copy(trk_hitidx.begin(), trk_hitidx.end(),inputtrk.hitidx);
//   ///*AHIT*/ inputhit = {
//   auto hit_pos = std::initializer_list<float>({-20.7824649810791, -12.24150276184082, 57.8067626953125});
//   auto hit_cov = std::initializer_list<float>({2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945});
//   
//    std::copy(hit_pos.begin(), hit_pos.end(),inputhit.pos);
//    std::copy(hit_cov.begin(), hit_cov.end(),inputhit.cov);

//   printf("track in pos: %f, %f, %f \n", inputtrk->par[0], inputtrk->par[1], inputtrk->par[2]);
//   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk->cov[SymOffsets66(PosInMtrx(0,0,6))],
//                                               inputtrk->cov[SymOffsets66(PosInMtrx(1,1,6))],
//                                               inputtrk->cov[SymOffsets66(PosInMtrx(2,2,6))]);
//   printf("hit in pos: %f %f %f \n", inputhit->pos[0], inputhit->pos[1], inputhit->pos[2]);
   printf("track in pos: %f, %f, %f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2]);
   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66(PosInMtrx(0,0,6))],
                                               inputtrk.cov[SymOffsets66(PosInMtrx(1,1,6))],
                                               inputtrk.cov[SymOffsets66(PosInMtrx(2,2,6))]);
   printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   //ALLTRKS* trk;// = new ALLTRKS;
   //ALLHITS* hit;// = new ALLHITS;
   //ALLTRKS* outtrk;// = new ALLTRKS;
   //cudaMallocManaged((void**)&trk,sizeof(ALLTRKS));
   //cudaMallocManaged((void**)&hit,sizeof(ALLHITS));
   //cudaMallocManaged((void**)&outtrk,sizeof(ALLTRKS));
   ALLTRKS* trk = prepareTracks(inputtrk);
   ALLHITS* hit = prepareHits(inputhit);

   printf("done preparing!\n");
   
   //ALLTRKS* outtrk = (ALLTRKS*) malloc(sizeof(ALLTRKS));
  #if USE_GPU
   ALLTRKS* outtrk;
   //prefetch(trk,hit,outtrk);
   cudaMallocManaged((void**)&outtrk,sizeof(ALLTRKS));
   int device = -1;
   cudaGetDevice(&device);
   //const int num_streams = 2;
   //cudaStream_t streams[num_streams];
   //for(int i =0; i<num_streams;i++){
   //cudaStreamCreate(&streams[i]);
   cudaMemPrefetchAsync(trk,sizeof(ALLTRKS), device,NULL);
   cudaMemPrefetchAsync(hit,sizeof(ALLHITS), device,NULL);
   cudaMemPrefetchAsync(outtrk,sizeof(ALLTRKS), device,NULL);
   //}
#else
   ALLTRKS* outtrk = (ALLTRKS*) malloc(sizeof(ALLTRKS));
#endif
   // for (size_t ie=0;ie<nevts;++ie) {
   //   for (size_t it=0;it<ntrks;++it) {
   //     printf("ie=%lu it=%lu\n",ie,it);
   //     printf("hx=%f\n",x(&hit,ie,it));
   //     printf("hy=%f\n",y(&hit,ie,it));
   //     printf("hz=%f\n",z(&hit,ie,it));
   //     printf("tx=%f\n",x(&trk,ie,it));
   //     printf("ty=%f\n",y(&trk,ie,it));
   //     printf("tz=%f\n",z(&trk,ie,it));
   //   }
   // }

   long start, end;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
#if USEACC
#pragma acc parallel loop
   for (size_t ie=0;ie<nevts;++ie) { // loop over events
     for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
       //
       const MPTRK* btracks = bTk(trk, ie, ib);
       const MPHIT* bhits = bHit(hit, ie, ib);
       MPTRK* obtracks = bTk(outtrk, ie, ib);
       propagateToZ(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
    }
  }
#else
#if USE_GPU
	GPUsequence1(trk,hit,outtrk);
       //const MPTRK* btracks = bTk(trk, ie, ib);
       //const MPHIT* bhits = bHit(hit, ie, ib);
       //MPTRK* obtracks = bTk(outtrk, ie, ib);
       //GPUsequence(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
       ////GPUpropagateToZ<<<1,1>>>(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
#else   
#pragma omp parallel for
   for (size_t ie=0;ie<nevts;++ie) { // loop over events
#pragma omp parallel for
     for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
       //
//#if USE_GPU
//       //GPUSequence(trk,hit,outtrk,ie,ib);
//       //MP6x6F errorProp_d;
//       //cudaMallocManaged((void**)&errorProp_d,sizeof(MP6x6F));
//
//       const MPTRK* btracks = bTk(trk, ie, ib);
//       const MPHIT* bhits = bHit(hit, ie, ib);
//       MPTRK* obtracks = bTk(outtrk, ie, ib);
//       GPUsequence(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
//       //GPUpropagateToZ<<<1,1>>>(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
//#else   
       const MPTRK* btracks = bTk(trk, ie, ib);
       const MPHIT* bhits = bHit(hit, ie, ib);
       MPTRK* obtracks = bTk(outtrk, ie, ib);
       propagateToZ(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
//#endif
    }
  }
#endif
#endif
   gettimeofday(&timecheck, NULL);
   end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   // for (size_t ie=0;ie<nevts;++ie) {
   //   for (size_t it=0;it<ntrks;++it) {
   //     printf("ie=%lu it=%lu\n",ie,it);
   //     printf("tx=%f\n",x(&outtrk,ie,it));
   //     printf("ty=%f\n",y(&outtrk,ie,it));
   //     printf("tz=%f\n",z(&outtrk,ie,it));
   //   }
   // }
   
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks, (end-start)*0.001, (end-start)*0.001/(nevts*ntrks));

   float avgx = 0, avgy = 0, avgz = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
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

#if USE_GPU
   cudaFree(&trk);
   cudaFree(&hit);
   cudaFree(&outtrk);
#else
//   free(&trk);
//   free(&hit);
//   free(&outtrk);

#endif
   return 0;
}
