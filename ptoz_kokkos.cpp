
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// kokkos
#include <Kokkos_Core.hpp>
// our typedefs curtisy of the examples
#include "kokkos_config.h"
#include "kokkos_gemm.h"
#include "ptoz_data.h"

#define AVAL 5.
#define BVAL 6.
#define TOL 0.0001



MPHIT* prepareHits(AHIT inputhit)


size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

size_t SymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

size_t SymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}

void read_input(int argc, char* argv[], 
                int &N, int &M, int &S, int &nrepeat, 
                int &batch_size, int &nbatches, bool &use_batches);
float randn(float mu, float sigma);


int main( int argc, char* argv[] )
{

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

  MPTRK* trk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
  prepareTracks(inputtrk, trk);
  MPHIT* hit = (MPHIT*) malloc(nevts*nb*sizeof(MPHIT));
  prepareHits(inputhit, hit);

  printf("done preparing!\n");

  MPTRK* outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      new(outtrk[ib + nb*ie].par)    ViewVectorMP("par", 6, bsize);      // batch of len 6 vectors
      new(outtrk[ib + nb*ie].cov)    ViewMatrixMP("cov", 6, 6, bsize);   // 6x6 symmetric batch matrix
      new(outtrk[ib + nb*ie].q)      ViewVectorINT("q", bsize);          // bsize array of int
      new(outtrk[ib + nb*ie].hitidx) ViewVectorINT("hidx", 22);          // unused? array len 22 of int
    }
  }


  long start, end;
  struct timeval timecheck;

  gettimeofday(&timecheck, NULL);
  start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

  for(itr=0; itr<NITER; itr++) {
  
  for (size_t ie=0;ie<nevts;++ie) { // loop over events
    for (size_t ib=0;ib<nb;++ib) { // loop over bunches of tracks

      const MPTRK* btracks = bTk(trk, ie, ib); // bTk picks out a specific track based on the event and bunch
      const MPHIT* bhits = bHit(hit, ie, ib);
      MPTRK* obtracks = bTk(outtrk, ie, ib);

      propagateToZ(&(*btracks).cov,  // MP6x6SF
                   &(*btracks).par,  // MP6F
                   &(*btracks).q,    // MP1I
                   &(*bhits).pos,    // MP3F
                   &(*obtracks).cov, // MP6x6SF
                   &(*obtracks).par  // MP6F
                   );

    } // nb
  } // evnts
  } // end of itr loop
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

  // TODO make sure these loops still make sense
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


  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      delete outtrk[ib + nb*ie].par;
      delete outtrk[ib + nb*ie].cov;
      delete outtrk[ib + nb*ie].q;
      delete outtrk[ib + nb*ie].hitidx;

      delete trk[ib + nb*ie].par;
      delete trk[ib + nb*ie].cov;
      delete trk[ib + nb*ie].q;
      delete trk[ib + nb*ie].hitidx;

      delete hit[ib + nb*ie].par;
      delete hit[ib + nb*ie].cov;
    }
  }
  free(trk);
  free(hit);
  free(outtrk);

  return 0;
}




// take the one track defined in main and make a bunch of "smeared" copies
// bsize is block size defined in ptoz_data.h
// nb is number of blocks
// TODO adjust so everything is a full matrix
void prepareTracks(ATRK inputtrk, MPTRK* &result) { // TODO the type on result is wrong

  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      new(result[ib + nb*ie].par)    ViewVectorMP("par", 6, bsize);      // batch of len 6 vectors
      new(result[ib + nb*ie].cov)    ViewMatrixMP("cov", 6, 6, bsize);   // 6x6 symmetric batch matrix
      new(result[ib + nb*ie].q)      ViewVectorINT("q", bsize);          // bsize array of int
      new(result[ib + nb*ie].hitidx) ViewVectorINT("hidx", 22);          // unused? array len 22 of int
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
  return result;
}

MPHIT* prepareHits(AHIT inputhit, MPHIT* result) {

  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      new(result[ib + nb*ie].pos)    ViewVectorMP("pos", 3, bsize);      // batch of len 3 vectors
      new(result[ib + nb*ie].cov)    ViewMatrixMP("cov", 6, 6, bsize);   // 6x6 symmetric batch matrix
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
        result[ib + nb*ie].cov[i][j][it] = (1+smear*randn(0,1))*inputtrk.cov[SymOffsets66(PosInMtrx(i,j,6))];
      
      } // bsize
    } // nb
  } // nevts
  return result;
}


void propagateToZ(const MP6x6SF* inErr, // input covariance
                  const MP6F* inPar,    // input parameters/state
                  const MP1I* inChg,    // input q from track
                  const MP3F* msP,      // input parameters from hit?
                  MP6x6SF* outErr,      // output covariance
                  MP6F* outPar) {       // output parameters/state
  //
  MP6x6F errorProp, temp; // TODO make views
#pragma omp simd
  for (size_t it=0;it<bsize;++it) { 
    const float zout = msP[Z_IND][it];
    const float k = inChg[it]*100/3.8;
    const float deltaZ = zout - inPar[Z_IND][it];
    const float pt = 1./inPar[IPT_IND][it];
    const float cosP = cosf(inPar[PHI_IND][it]);
    const float sinP = sinf(inPar[PHI_IND][it]);
    const float cosT = cosf(inPar[THETA_IND][it]);
    const float sinT = sinf(inPar[THETA_IND][it]);
    const float pxin = cosP*pt;
    const float pyin = sinP*pt;
    const float alpha = deltaZ*sinT*inPar[PHI_IND][it]/(cosT*k);
    const float sina = sinf(alpha); // this can be approximated;
    const float cosa = cosf(alpha); // this can be approximated;

    // array of state
    outPar[X_IND][it]     = inPar[X_IND][it] + k*(pxin*sina - pyin*(1.-cosa));
    outPar[Y_IND][it]     = inPar[Y_IND][it] + k*(pyin*sina + pxin*(1.-cosa));
    outPar[Z_IND][it]     = zout;
    outPar[IPT_IND][it]   = inPar[PHI_IND][it];
    outPar[PHI_IND][it]   = inPar[PHI_IND][it]+alpha;
    outPar[THETA_IND][it] = inPar[THETA_IND][it];
    
    const float sCosPsina = sinf(cosP*sina);
    const float cCosPsina = cosf(cosP*sina);
    
    for (size_t i=0;i<6;++i) errorProp[i][i][it] = 1.;
    errorProp[2][0][it] = errorProp[0][2][it] = cosP*sinT*(sinP*cosa*sCosPsina-cosa)/cosT;
    errorProp[3][0][it] = errorProp[0][3][it] = cosP*sinT*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*inPar[PHI_IND][it])-k*(cosP*sina-sinP*(1.-cCosPsina))/(inPar[PHI_IND][it]*inPar[PHI_IND][it]);
    errorProp[4][0][it] = errorProp[0][4][it] = (k/inPar[PHI_IND][it])*(-sinP*sina+sinP*sinP*sina*sCosPsina-cosP*(1.-cCosPsina));
    errorProp[5][0][it] = errorProp[0][5][it] = cosP*deltaZ*cosa*(1.-sinP*sCosPsina)/(cosT*cosT);
    errorProp[2][1][it] = errorProp[1][2][it] = cosa*sinT*(cosP*cosP*sCosPsina-sinP)/cosT;
    errorProp[3][1][it] = errorProp[1][3][it] = sinT*deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*inPar[PHI_IND][it])-k*(sinP*sina+cosP*(1.-cCosPsina))/(inPar[PHI_IND][it]*inPar[PHI_IND][it]);
    errorProp[4][1][it] = errorProp[1][4][it] = (k/inPar[PHI_IND][it])*(-sinP*(1.-cCosPsina)-sinP*cosP*sina*sCosPsina+cosP*sina);
    errorProp[5][1][it] = errorProp[1][5][it] = deltaZ*cosa*(cosP*cosP*sCosPsina+sinP)/(cosT*cosT);
    errorProp[2][4][it] = errorProp[4][2][it] = -inPar[PHI_IND][it]*sinT/(cosT*k);
    errorProp[3][4][it] = errorProp[4][3][it] = sinT*deltaZ/(cosT*k);
    errorProp[5][4][it] = errorProp[4][5][it] = inPar[PHI_IND][it]*deltaZ/(cosT*cosT*k);
  }
  //
  // TODO make these gemms
  MultHelixPropEndcap(&errorProp, inErr, &temp);
  MultHelixPropTranspEndcap(&errorProp, &temp, outErr);
}



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



