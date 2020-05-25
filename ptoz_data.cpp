/*
 * matrix_batch.h
 
 * Brian J Gravelle
 * ix.cs.uoregon.edu/~gravelle
 * gravelle@cs.uoregon.edu

 * this is code for batched matrix operations on p2z
 * the key is that the matrices are held in a 3D array
 *   - the 3rd dimension is which matrix it belongs to
 *   - this allows the operations to be SIMDized 

 * See LICENSE file for licensing information and boring legal stuff

 * If by some miricale you find this software useful, thanks are accepted in
 * the form of chocolate, coffee, or introductions to potential employers.

*/ 

#include "ptoz_data.h"

void allocate_MPTRK(struct MPTRK &trk) {

  trk.par = (float**)malloc(sizeof(float*)*6);      // batch of len 6 vectors
  trk.cov = (float***)malloc(sizeof(float**)*6);   // 6x6 symmetric batch matrix
  for (int i = 0; i < 6; i++){
    trk.par[i] = (float*)malloc(sizeof(float)*bsize);      // batch of len 6 vectors
    trk.cov[i] = (float**)malloc(sizeof(float*)*6);   // 6x6 symmetric batch matrix
    for (int j = 0; j < bsize; j++){
      trk.cov[i][j] = (float*)malloc(sizeof(float)*bsize);
    }
  }
  trk.q = (int*)malloc(sizeof(float)*bsize);         // bsize array of int
  trk.hitidx = (int*)malloc(sizeof(int)*22);            // unused? array len 22 of int

}
void free_MPTRK(struct MPTRK &trk) {

  for (int i = 0; i < 6; i++){
    for (int j = 0; j < bsize; j++){
      free(trk.cov[i][j]);
    }
    free(trk.par[i]);
    // free(trk.cov[i]);
  }
  // free(trk.par);
  // free(trk.cov);
  // free(trk.q);
  // free(trk.hitidx);

}

void allocate_MPHIT(struct MPHIT &hit) {

  hit.pos = (float**)malloc(sizeof(float*)*6);     // batch of len 6 vectors
  hit.cov = (float***)malloc(sizeof(float**)*6);   // 6x6 symmetric batch matrix
  for (int i = 0; i < 6; i++){
    hit.pos[i] = (float*)malloc(sizeof(float)*bsize); // batch of len 6 vectors
    hit.cov[i] = (float**)malloc(sizeof(float*)*6);   // 6x6 symmetric batch matrix
    for (int j = 0; j < bsize; j++){
      hit.cov[i][j] = (float*)malloc(sizeof(float)*bsize);
    }
  }

}

void free_MPHIT(struct MPHIT &hit) {

  for (int i = 0; i < 6; i++){
    for (int j = 0; j < bsize; j++){
      free(hit.cov[i][j]);
    }
    free(hit.pos[i]);
    // free(hit.cov[i]);
  }
  // free(hit.pos);
  // free(hit.cov);

}

// TODO make all these check for success
void allocate_MatrixMKL(struct MatrixMKL &mat, int num_bats) {
  
  mat.vals = (float**) mkl_malloc(bsize*num_bats*sizeof(float*),64); // batches of mats
  for (int i = 0; i < bsize*num_bats; i++){
    mat.vals[i] = (float*) mkl_malloc(6*6*sizeof(float),64);
  }
  mat.num_mats = bsize;  // number of matrices per batch
  mat.num_bats = num_bats;  // number of batches

}
void free_MatrixMKL(struct MatrixMKL &mat, int num_bats) {
  
  for (int i = 0; i < bsize*num_bats; i++){
    mkl_free(mat.vals[i]);
  }
  // mkl_free(mat.vals);

}



void allocate_MKLTRK(struct MKLTRK &trk, int num_bats) {
  
  trk.cov = (float**) mkl_malloc(bsize*num_bats*sizeof(float*),64); // batches of mats
  trk.par = (float**) mkl_malloc(bsize*num_bats*sizeof(float*),64); // batches of vecs
  trk.q   = (int*) mkl_malloc(bsize*num_bats*sizeof(int),64);
  for (int i = 0; i < bsize*num_bats; i++){
    trk.cov[i] = (float*) mkl_malloc(6*6*sizeof(float),64);
    trk.par[i] = (float*) mkl_malloc(6*sizeof(float),64);
  }
  trk.num_mats = bsize;  // number of matrices per batch
  trk.num_bats = num_bats;  // number of batches

}

void free_MKLTRK(struct MKLTRK &trk, int num_bats) {
  
  for (int i = 0; i < bsize*num_bats; i++){
    mkl_free(trk.cov[i]);
    mkl_free(trk.par[i]);
  }
  // mkl_free(trk.cov);
  // mkl_free(trk.par);
  // mkl_free(trk.q);

}

void allocate_MKLHIT(struct MKLHIT &hit, int num_bats) {

  hit.pos = (float**) mkl_malloc(bsize*num_bats*sizeof(float*),64); // len 3 vec
  hit.cov = (float**) mkl_malloc(bsize*num_bats*sizeof(float*),64); // 6x6 mat
  for (int i = 0; i < bsize*num_bats; i++){
    hit.pos[i] = (float*) mkl_malloc(3*sizeof(float),64);
    hit.cov[i] = (float*) mkl_malloc(36*sizeof(float),64);
  }  
  hit.num_mats = bsize;  // number of matrices per batch
  hit.num_bats = num_bats;  // number of batches

}

void free_MKLHIT(struct MKLHIT &hit, int num_bats) {

  for (int i = 0; i < bsize*num_bats; i++){
    mkl_free(hit.pos[i]);
    mkl_free(hit.cov[i]);
  }  
  // mkl_free(hit.pos);
  // mkl_free(hit.cov);

}


void mkl_compact(float **A, float **B, float **C, 
                 float *_A, float *_B, float *_C, 
                 int num_mats) {

  int i,j,l;

  // float *_A, *_B, *_C;
  
  MKL_COMPACT_PACK mkl_format = mkl_get_format_compact();
  // MKL_INT mkl_size_compact    = mkl_sget_size_compact (6, 6, mkl_format, num_mats);

  mkl_sgepack_compact(MKL_ROW_MAJOR, 6, 6, A, 6, _A, 6, mkl_format, num_mats);
  mkl_sgepack_compact(MKL_ROW_MAJOR, 6, 6, B, 6, _B, 6, mkl_format, num_mats);
  // mkl_sgepack_compact(MKL_ROW_MAJOR, 6, 6, C, 6, _C, 6, mkl_format, num_mats);

  mkl_sgemm_compact(
    MKL_ROW_MAJOR,  // layout 
    MKL_NOTRANS,    // transpose A
    MKL_NOTRANS,    // transpose B
    6,              // A rows
    6,              // B cols
    6,              // A col B row
    1.0,            // alpha (scalar for A)
    _A,             // double* to A
    6,              // still have no idea
    _B,             // double* to B
    6,              // still have no idea
    0.0,            // beta (scalar for C)
    _C,             // double* to C
    6,              // still have no idea
    mkl_format,     // compact format from the mkl function
    num_mats        // total number of matrices
  );
  mkl_sgemm_compact(
    MKL_ROW_MAJOR,  // layout 
    MKL_NOTRANS,    // transpose A
    MKL_TRANS,      // transpose B
    6,              // A rows
    6,              // B cols
    6,              // A col B row
    1.0,            // alpha (scalar for A)
    _A,             // double* to A
    6,              // still have no idea
    _C,             // double* to B
    6,              // still have no idea
    0.0,            // beta (scalar for C)
    _B,             // double* to C
    6,              // still have no idea
    mkl_format,     // compact format from the mkl function
    num_mats        // total number of matrices
  );

  // mkl_sgeunpack_compact (MKL_ROW_MAJOR, 6, 6, A, 6, _A, 6, mkl_format, num_mats);
  // mkl_sgeunpack_compact (MKL_ROW_MAJOR, 6, 6, B, 6, _B, 6, mkl_format, num_mats);
  mkl_sgeunpack_compact (MKL_ROW_MAJOR, 6, 6, C, 6, _B, 6, mkl_format, num_mats);


}



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

