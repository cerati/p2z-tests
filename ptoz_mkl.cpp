
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#include "ptoz_data.h"

#define AVAL 5.
#define BVAL 6.
#define TOL 0.0001


void prepareTracks(ATRK inputtrk, MPTRK* &result);
void separateTracks(MPTRK* &mp_in, MKLTRK &mkl_out);
void prepareHits(AHIT inputhit, MPHIT* &result);
void separateHits(MPHIT* &mp_in, MKLHIT &mkl_out);
void convertOutput(MKLTRK &mkl_in, MPTRK* &mp_out);

void propagateToZ(const MKLTRK &inTrks, const MKLHIT &inHits, MKLTRK &outTrks, 
                  MatrixMKL &errorProp,
                  float *_A, float *_B, float *_C,
                  int _nevts_nb, int _bsize);
void averageOutputs(MPTRK* &outtrk, MPHIT* &hit);

// void gemm(MatrixMP A, ViewMatrixMP B, ViewMatrixMP C);
// void gemm_T(MatrixMP A, ViewMatrixMP B, ViewMatrixMP C);

double get_time();

void read_input(int argc, char* argv[], 
                int &N, int &M, int &S, int &nrepeat, 
                int &batch_size, int &nbatches, bool &use_batches);
float randn(float mu, float sigma);


int main( int argc, char* argv[] )
{

#ifdef USE_CALI
cali_id_t thread_attr = cali_create_attribute("thread_id", CALI_TYPE_INT, CALI_ATTR_ASVALUE | CALI_ATTR_SKIP_EVENTS);
#pragma omp parallel
{
cali_set_int(thread_attr, omp_get_thread_num());
}
#endif


  int block_in = -1;
  if (argc > 1) block_in = std::atoi(argv[1]);

  int itr;
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

  printf("track in pos: %f, %f, %f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2]);
  printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66(PosInMtrx(0,0,6))],
                                       inputtrk.cov[SymOffsets66(PosInMtrx(1,1,6))],
                                       inputtrk.cov[SymOffsets66(PosInMtrx(2,2,6))]);
  printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);

  printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
  printf("NITER=%d\n", NITER);
  printf("bsize=%d\n", bsize);


  double start_t, prep_t, convert_in_t, p2z_t, convert_out_t, end_t;

  start_t = get_time();

  MPTRK *trk;
  prepareTracks(inputtrk, trk);
  MPHIT *hit;
  prepareHits(inputhit, hit);
  MPTRK* outtrk;
  outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
  for (size_t ie=0;ie<nevts;++ie) 
    for (size_t ib=0;ib<nb;++ib) 
      allocate_MPTRK(outtrk[ib + nb*ie]);    
  
  prep_t = get_time();

  MKLTRK all_tracks;
  MKLHIT all_hits;
  MKLTRK all_out;
  allocate_MKLTRK(all_tracks, nevts*nb);
  allocate_MKLHIT(all_hits, nevts*nb);
  allocate_MKLTRK(all_out, nevts*nb);

  //TODO check these function calls
  separateTracks(trk, all_tracks);
  separateHits(hit, all_hits);

  MatrixMKL errorProp;
  allocate_MatrixMKL(errorProp, nevts*nb);

  float *_A, *_B, *_C;
  MKL_COMPACT_PACK mkl_format = mkl_get_format_compact();
  MKL_INT mkl_size_compact    = mkl_sget_size_compact (6, 6, mkl_format, nevts*nb*bsize);
  _A = (float*) mkl_malloc(mkl_size_compact,64);
  _B = (float*) mkl_malloc(mkl_size_compact,64);
  _C = (float*) mkl_malloc(mkl_size_compact,64);

  printf("done preparing!\n");
  
  convert_in_t = get_time();
  
  int block_size = 500;  // must divide into 60,000, 500 -> 2e-6s
  // int block_size = 1500; // must divide into 60,000
  // int block_size = 60000;    // must divide into 60,000
  if (block_in > 0)
    block_size = block_in;
  // int block_size = nevts*nb;
  int b = 0;
  MKLTRK block_tracks;
  MKLHIT block_hits;
  MKLTRK block_out;
  MatrixMKL blockProp;

  for(itr=0; itr<NITER; itr++) {
  
    #pragma omp parallel for
    for(b = 0; b < nevts*nb*bsize; b+=block_size*bsize) {
      block_tracks.par = &(all_tracks.par[b]);
      block_tracks.cov = &(all_tracks.cov[b]);
      block_tracks.q   = &(all_tracks.q[b]);

      block_hits.pos   = &(all_hits.pos[b]);
      block_hits.cov   = &(all_hits.cov[b]);

      block_out.par    = &(all_out.par[b]);
      block_out.cov    = &(all_out.cov[b]);
      block_out.q      = &(all_out.q[b]);

      blockProp.vals   = &(errorProp.vals[b]);

      propagateToZ(block_tracks, block_hits, block_out, blockProp, 
                  _A, _B, _C,
                  block_size, bsize);
    }

  } // end of itr loop

  p2z_t = get_time();

  // TODO allout -> outtrk
  convertOutput(all_out, outtrk);
  // for (size_t ie=0;ie<nevts;++ie) {
  //   for (size_t it=0;it<ntrks;++it) {
  //     printf("ie=%lu it=%lu\n",ie,it);
  //     printf("tx=%f\n",x(&outtrk,ie,it));
  //     printf("ty=%f\n",y(&outtrk,ie,it));
  //     printf("tz=%f\n",z(&outtrk,ie,it));
  //   }
  // }

  convert_out_t = get_time();

  averageOutputs(outtrk, hit);

  end_t = get_time();

  printf("done ntracks =%i \n", nevts*ntrks);
  printf("done niter   =%i \n", NITER);
  printf("total tracks =%i \n", nevts*ntrks*NITER);
  printf("block size   =%i \n", block_size);
  printf("Total time   =%f \n", end_t - start_t);
  printf("MP prep time =%e \n", prep_t - start_t);
  printf("MKL convert  =%e \n", (convert_in_t - prep_t) + (convert_out_t - p2z_t));
  printf("p2z time     =%e \n", p2z_t - convert_in_t);
  printf("Time / track =%e \n", (p2z_t - convert_in_t) / (float)(nevts*ntrks) );
  

  // for (size_t ie=0;ie<nevts;++ie) {
  //   for (size_t ib=0;ib<nb;++ib) {
  //     free_MPTRK(trk[ib + nb*ie]);
  //     free_MPHIT(hit[ib + nb*ie]);
  //     free_MPTRK(outtrk[ib + nb*ie]);
  //   }
  // }

  mkl_free(_A);
  mkl_free(_B);
  mkl_free(_C);

  free_MatrixMKL(errorProp, nevts*nb);

  free_MKLTRK(all_tracks, nevts*nb);
  free_MKLHIT(all_hits, nevts*nb);
  free_MKLTRK(all_out, nevts*nb);


  return 0;
}



// take the one track defined in main and make a bunch of "smeared" copies
// bsize is block size defined in ptoz_data.h
// nb is number of blocks
// TODO adjust so everything is a full matrix
void prepareTracks(ATRK inputtrk, MPTRK* &result) {

  result = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));

  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      allocate_MPTRK(result[ib + nb*ie]);
    }
  }

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {

  //par
  for (size_t ip=0;ip<6;++ip) {
    result[ib + nb*ie].par[ip][it] = (1+smear*randn(0,1))*inputtrk.par[ip];
  }
  //cov
  for (size_t i=0;i<6;++i)
    for (size_t j=0;j<6;++j)
      result[ib + nb*ie].cov[i][j][it] = (1+smear*randn(0,1))*inputtrk.cov[SymOffsets66(PosInMtrx(i,j,6))];
  //q
  result[ib + nb*ie].q[it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      
      } // block loop
    } // nb
  } // nevts
}


void separateTracks(MPTRK* &mp_in, MKLTRK &mkl_out) {

  for (size_t ie=0;ie<nevts;++ie) {
  for (size_t ib=0;ib<nb;++ib) {
    int batch = ib + nb*ie;

    // q int batch
    for (size_t it=0;it<bsize;++it) 
      mkl_out.q[batch*bsize+it] = mp_in[batch].q[it];

    // par length 6 vector batch
    for (size_t ip=0;ip<6;++ip) {
      for (size_t it=0;it<bsize;++it) 
        mkl_out.par[batch*bsize+it][ip] = mp_in[batch].par[ip][it];
    }

    // cov 6x6 matrix batch
    for (size_t i=0;i<6;++i) {
      for (size_t j=0;j<6;++j) {
        for (size_t it=0;it<bsize;++it) 
          mkl_out.cov[batch*bsize+it][i*6+j] = mp_in[batch].cov[i][j][it];
      }
    }
      
  } // nb
  } // nevts

}

void convertOutput(MKLTRK &mkl_in, MPTRK* &mp_out) {

  for (size_t ie=0;ie<nevts;++ie) {
  for (size_t ib=0;ib<nb;++ib) {
    int batch = ib + nb*ie;

    // q int batch
    for (size_t it=0;it<bsize;++it) 
      mp_out[batch].q[it] = mkl_in.q[batch*bsize+it];

    // par length 6 vector batch
    for (size_t ip=0;ip<6;++ip) {
      for (size_t it=0;it<bsize;++it) 
        mp_out[batch].par[ip][it] = mkl_in.par[batch*bsize+it][ip];
    }

    // cov 6x6 matrix batch
    for (size_t i=0;i<6;++i) {
      for (size_t j=0;j<6;++j) {
        for (size_t it=0;it<bsize;++it) 
          mp_out[batch].cov[i][j][it] = mkl_in.cov[batch*bsize+it][i*6+j];
      }
    }
      
  } // nb
  } // nevts

  // Kokkos needed this to happen twice so we'll leave it for now
  for (size_t ie=0;ie<nevts;++ie) {
  for (size_t ib=0;ib<nb;++ib) {
    int batch = ib + nb*ie;

    // q int batch
    for (size_t it=0;it<bsize;++it) 
      mp_out[batch].q[it] = mkl_in.q[batch*bsize+it];

    // par length 6 vector batch
    for (size_t ip=0;ip<6;++ip) {
      for (size_t it=0;it<bsize;++it) 
        mp_out[batch].par[ip][it] = mkl_in.par[batch*bsize+it][ip];
    }

    // cov 6x6 matrix batch
    for (size_t i=0;i<6;++i) {
      for (size_t j=0;j<6;++j) {
        for (size_t it=0;it<bsize;++it) 
          mp_out[batch].cov[i][j][it] = mkl_in.cov[batch*bsize+it][i*6+j];
      }
    }
      
  } // nb
  } // nevts

}


void prepareHits(AHIT inputhit, MPHIT* &result) {

  result = (MPHIT*) malloc(nevts*nb*sizeof(MPHIT));

  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      allocate_MPHIT(result[ib + nb*ie]);
    }
  } 

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
    
    //pos
    for (size_t ip=0;ip<3;++ip) {
      result[ib + nb*ie].pos[ip][it] = (1+smear*randn(0,1))*inputhit.pos[ip];
    }
    //cov
    for (size_t i=0;i<6;++i)
      for (size_t j=0;j<6;++j)
        result[ib + nb*ie].cov[i][j][it] = (1+smear*randn(0,1))*inputhit.cov[SymOffsets66(PosInMtrx(i,j,6))];
      
      } // bsize
    } // nb
  } // nevts
}


void separateHits(MPHIT* &mp_in, MKLHIT &mkl_out) {

  for (size_t ie=0;ie<nevts;++ie) {
  for (size_t ib=0;ib<nb;++ib) {
    int batch = ib + nb*ie;

    // pos length 3 vector batch
    for (size_t ip=0;ip<3;++ip) {
      for (size_t it=0;it<bsize;++it) 
        mkl_out.pos[batch*bsize+it][ip] = mp_in[batch].pos[ip][it];
    }

    // cov 6x6 matrix batch
    for (size_t i=0;i<6;++i) {
      for (size_t j=0;j<6;++j) {
        for (size_t it=0;it<bsize;++it) 
          mkl_out.cov[batch*bsize+it][i*6+j] = mp_in[batch].cov[i][j][it];

      }
    }
      
  } // nb
  } // nevts

}

// MP version
// example:
//      propagateToZ(btracks.cov,  // MP6x6SF
//                   btracks.par,  // MP6F
//                   btracks.q,    // MP1I
//                   bhits.pos,    // MP3F
//                   obtracks.cov, // MP6x6SF
//                   obtracks.par  // MP6F
//                   );
// void propagateToZ(const ViewMatrixMP inErr,  // input covariance
//                   const ViewVectorMP inPar,  // input parameters/state
//                   const ViewVectorINT inChg, // input q from track
//                   const ViewVectorMP msP,    // input parameters from hit?
//                   ViewMatrixMP outErr,       // output covariance
//                   ViewVectorMP outPar) {     // output parameters/state
  //


void propagateToZ(const MKLTRK &inTrks, const MKLHIT &inHits, MKLTRK &outTrks, 
                  MatrixMKL &errorProp,
                  float *_A, float *_B, float *_C,
                  int _nevts_nb, int _bsize) { 

#ifdef USE_CALI
CALI_CXX_MARK_FUNCTION;
#endif

  // #pragma omp parallel for
  // for (size_t ie=0;ie<_nevts;++ie) { // combined these two loop over batches
  // for (size_t ib=0;ib<_nb;++ib) { 
  // size_t batch = ib + _nb*ie;
  for (size_t batch=0; batch < _nevts_nb; ++batch) { // combined these two loop over batches
  #pragma ivdep
  #pragma simd
  for (size_t it=0;it<_bsize;++it) { 

    const float zout = inHits.pos[batch*_bsize+it][Z_IND]; 
    const float k = inTrks.q[batch*_bsize+it]*100/3.8; 
    const float deltaZ = zout - inTrks.par[batch*_bsize+it][Z_IND]; 
    const float pt = 1./inTrks.par[batch*_bsize+it][IPT_IND]; 
    const float cosP = cosf( inTrks.par[batch*_bsize+it][PHI_IND] ); 
    const float sinP = sinf( inTrks.par[batch*_bsize+it][PHI_IND] ); 
    const float cosT = cosf( inTrks.par[batch*_bsize+it][THETA_IND] ); 
    const float sinT = sinf( inTrks.par[batch*_bsize+it][THETA_IND] ); 
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float alpha = deltaZ*sinT*inTrks.par[batch*_bsize+it][IPT_IND]/(cosT*k); 
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;

    // array of state
    outTrks.par[batch*_bsize+it][X_IND]     = inTrks.par[batch*_bsize+it][X_IND] + k*(pxin*sina - pyin*(1.-cosa));
    outTrks.par[batch*_bsize+it][Y_IND]     = inTrks.par[batch*_bsize+it][Y_IND] + k*(pyin*sina + pxin*(1.-cosa));
    outTrks.par[batch*_bsize+it][Z_IND]     = zout;
    outTrks.par[batch*_bsize+it][IPT_IND]   = inTrks.par[batch*_bsize+it][IPT_IND];
    outTrks.par[batch*_bsize+it][PHI_IND]   = inTrks.par[batch*_bsize+it][PHI_IND]+alpha;
    outTrks.par[batch*_bsize+it][THETA_IND] = inTrks.par[batch*_bsize+it][THETA_IND];
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
    
    for (size_t i=0;i<6;++i) errorProp.vals[batch*_bsize+it][i*6+i] = 1.;
    //there are two cause we're doing symmetry
    errorProp.vals[batch*_bsize+it][2*6+0] = errorProp.vals[batch*_bsize+it][0*6+2] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
    errorProp.vals[batch*_bsize+it][3*6+0] = errorProp.vals[batch*_bsize+it][0*6+3] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*inTrks.par[batch*_bsize+it][IPT_IND])-k*(cosP*sina-sinP*(1.-cCosPsina))/(inTrks.par[batch*_bsize+it][IPT_IND]*inTrks.par[batch*_bsize+it][IPT_IND]);
    errorProp.vals[batch*_bsize+it][4*6+0] = errorProp.vals[batch*_bsize+it][0*6+4] = (k/inTrks.par[batch*_bsize+it][IPT_IND])*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp.vals[batch*_bsize+it][5*6+0] = errorProp.vals[batch*_bsize+it][0*6+5] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
    errorProp.vals[batch*_bsize+it][2*6+1] = errorProp.vals[batch*_bsize+it][1*6+2] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
    errorProp.vals[batch*_bsize+it][3*6+1] = errorProp.vals[batch*_bsize+it][1*6+3] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*inTrks.par[batch*_bsize+it][IPT_IND])-k*(sinP*sina+cosP*(1.-cCosPsina))/(inTrks.par[batch*_bsize+it][IPT_IND]*inTrks.par[batch*_bsize+it][IPT_IND]);
    errorProp.vals[batch*_bsize+it][4*6+1] = errorProp.vals[batch*_bsize+it][1*6+4] = (k/inTrks.par[batch*_bsize+it][IPT_IND])*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp.vals[batch*_bsize+it][5*6+1] = errorProp.vals[batch*_bsize+it][1*6+5] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
    errorProp.vals[batch*_bsize+it][2*6+4] = errorProp.vals[batch*_bsize+it][4*6+2] = -inTrks.par[batch*_bsize+it][IPT_IND]*sinT/(cosT*k);
    errorProp.vals[batch*_bsize+it][3*6+4] = errorProp.vals[batch*_bsize+it][4*6+3] = sinT*deltaZ/(cosT*k);
    errorProp.vals[batch*_bsize+it][5*6+4] = errorProp.vals[batch*_bsize+it][4*6+5] = inTrks.par[batch*_bsize+it][IPT_IND]*deltaZ/(cosT*cosT*k);

  } // bsize

  // TODO make blocking work
  // MKL_COMPACT_PACK mkl_format = mkl_get_format_compact();
  // MKL_INT size = mkl_sget_size_compact (6, 6, mkl_format, bsize) / sizeof(float);
  // mkl_compact(&(errorProp.vals[batch*bsize]), &(inTrks.cov[batch*bsize]), &(outTrks.cov[batch*bsize]),   
  //           &(_A[batch*size]), &(_B[batch*size]), &(_C[batch*size]), bsize);

// } // nb
}  // nevts

  mkl_compact(errorProp.vals, inTrks.cov, outTrks.cov,   
            _A, _B, _C, _nevts_nb*_bsize);
// for (size_t ie=0;ie<nevts;++ie) { // combined these two loop over batches
// for (size_t ib=0;ib<nb;++ib) { 
//   size_t batch = ib + nb*ie;
  // for ( int i = 0; i < 6; ++i ) {
  //   for ( int j = 0; j < 6; ++j ) {

  //     #pragma ivdep
  //     #pragma simd
  //     for ( int it = 0; it < bsize; ++it ) 
  //       temp.vals[batch*bsize+it][i*6+j] = 0.0;

  //     for ( int k = 0; k < 6; ++k ) {
  //       #pragma ivdep
  //       #pragma simd
  //       for ( int it = 0; it < bsize; ++it ) {
  //         temp.vals[batch*bsize+it][i*6+j] += errorProp.vals[batch*bsize+it][i*6+k] * inTrks.cov[batch*bsize+it][k*6+j];
  //       }
  //     }
      
  //   }
  // } //gemm

  //gemm with B transposed
//   for ( int i = 0; i < 6; ++i ) {
//     for ( int j = 0; j < 6; ++j ) {

//       #pragma ivdep
//       #pragma simd
//       for ( int it = 0; it < bsize; ++it ) 
//         outTrks.cov[batch*bsize+it][i*6+j] = 0.0;

//       for ( int k = 0; k < 6; ++k ) {
//         #pragma ivdep
//         #pragma simd
//         for ( int it = 0; it < bsize; ++it ) {
//           outTrks.cov[batch*bsize+it][i*6+j] += errorProp.vals[batch*bsize+it][i*6+k] * temp.vals[batch*bsize+it][j*6+k];
//         }
//       }
      
//     }
//   } //gemmT


// } // nb
// }  // nevts

}


void averageOutputs(MPTRK* &outtrk, MPHIT* &hit) { 

  float avgx = 0, avgy = 0, avgz = 0;
  float avgdx = 0, avgdy = 0, avgdz = 0;
  for (size_t ie=0;ie<nevts;++ie) { // loop over events
    for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
      for (size_t it=0;it<bsize;++it) {
        float x_ = outtrk[ib + nb*ie].par[X_IND][it];
        float y_ = outtrk[ib + nb*ie].par[Y_IND][it];
        float z_ = outtrk[ib + nb*ie].par[Z_IND][it];
        avgx += x_;
        avgy += y_;
        avgz += z_;
        float hx_ = hit[ib + nb*ie].pos[X_IND][it];
        float hy_ = hit[ib + nb*ie].pos[Y_IND][it];
        float hz_ = hit[ib + nb*ie].pos[Z_IND][it];
        avgdx += (x_-hx_)/x_;
        avgdy += (y_-hy_)/y_;
        avgdz += (z_-hz_)/z_;
      }
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
  for (size_t ie=0;ie<nevts;++ie) { // loop over events
    for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks
      for (size_t it=0;it<bsize;++it) {
        float x_ = outtrk[ib + nb*ie].par[X_IND][it];
        float y_ = outtrk[ib + nb*ie].par[Y_IND][it];
        float z_ = outtrk[ib + nb*ie].par[Z_IND][it];
        stdx += (x_-avgx)*(x_-avgx);
        stdy += (y_-avgy)*(y_-avgy);
        stdz += (z_-avgz)*(z_-avgz);
        float hx_ = hit[ib + nb*ie].pos[X_IND][it];
        float hy_ = hit[ib + nb*ie].pos[Y_IND][it];
        float hz_ = hit[ib + nb*ie].pos[Z_IND][it];
        stddx += ((x_-hx_)/x_-avgdx)*((x_-hx_)/x_-avgdx);
        stddy += ((y_-hy_)/y_-avgdy)*((y_-hy_)/y_-avgdy);
        stddz += ((z_-hz_)/z_-avgdz)*((z_-hz_)/z_-avgdz);
      }
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
}


// TODO do we use these?
// void gemm(ViewMatrixMP A, ViewMatrixMP B, ViewMatrixMP C) {

//   for ( int i = 0; i < 6; ++i ) {
//     for ( int j = 0; j < 6; ++j ) {

//       for ( int b = 0; b < bsize; ++b ) 
//         C(i,j,b) = 0.0;

//       for ( int k = 0; k < 6; ++k ) {
//         for ( int b = 0; b < bsize; ++b ) {
//           C(i,j,b) += A(i,k,b) * B(k,j,b);
//         }
//       }
      
//     }
//   }

// }


// void gemm_T(ViewMatrixMP A, ViewMatrixMP B, ViewMatrixMP C) {

//   for ( int i = 0; i < 6; ++i ) {
//     for ( int j = 0; j < 6; ++j ) {

//       for ( int b = 0; b < bsize; ++b ) 
//         C(i,j,b) = 0.0;

//       for ( int k = 0; k < 6; ++k ) {
//         for ( int b = 0; b < bsize; ++b ) {
//           C(i,j,b) += A(i,k,b) * B(j,k,b);
//         }
//       }
      
//     }
//   }

// }
// 


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

double get_time() {
  struct timeval timecheck;
  gettimeofday( &timecheck, NULL );
  double time = ( 1.0 * timecheck.tv_sec ) + ( 1.0e-6 * timecheck.tv_usec );
  return time;
}

/* Copyright for the lovely input reader below
// It comes from the kokkos examples
// but we have lightly modified it to work with nice small matrices
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
void read_input(int argc, char* argv[], int &N, int &M, int &S, int &nrepeat, int &batch_size, int &nbatches, bool &use_batches) {

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = atoi( argv[ ++i ] );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = atoi( argv[ ++i ] );;
      printf( "  User M is %d\n", M );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-batch_size" ) == 0 ) {
      batch_size = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-nbatches" ) == 0 ) {
      nbatches = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-use_batches" ) == 0 ) {
      use_batches = true;
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      determines number of rows (default: 1024)\n" );
      printf( "  -Columns (-M) <int>:   determines number of columns (default: 1024)\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -batch_size <int>:     number of matrices per batch (default: 100)\n" );
      printf( "  -nbatches <int>:       number of batches (default: 100)\n" );
      printf( "  -use_batches:          perform batch based test\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = 1024;

  // If N is undefined, set it.
  if ( N == -1 ) N = 1024;

  S = M*N;

  printf( "  Total size S = %d N = %d M = %d\n", S, N, M );
  printf( "  Using %d batches of %d matrices\n", nbatches, batch_size );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }

}



