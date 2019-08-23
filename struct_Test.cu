#include <stdio.h>
#include <stdlib.h>


struct ATRK {
  float par[6];
  float cov[21];
  int q;
  int hitidx[22];
};

__global__ void test(ATRK inputtrk){
printf("in kernel\n");
printf("par: %f,%f,%f,%f,%f,%f\n",inputtrk.par[0],inputtrk.par[1],inputtrk.par[2],inputtrk.par[3],inputtrk.par[4],inputtrk.par[5]);
printf("second kernel\n");
}


int main (int argc, char* argv[]) {

   ATRK inputtrk;

cudaMallocManaged((void**) &inputtrk, sizeof(ATRK)); 
	inputtrk = {
     {-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975},
     {6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
      6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,                                                                                      0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348},                                                                                   1,
     {1, 0, 17, 16, 36, 35, 33, 34, 59, 58, 70, 85, 101, 102, 116, 117, 132, 133, 152, 169, 187, 202}
   };
printf("starting test\n");
//cudaMalloc((void**) &inputtrk, sizeof(ATRK)); 
test<<<1,1>>>(inputtrk);
cudaDeviceSynchronize();
printf("finished test\n");
}
