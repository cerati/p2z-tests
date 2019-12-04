#include <cstdio>
#include <cstdlib>

struct point  
{  
     double x,y;  
};

__global__ void MyFunc(point* d_a)  
{  
     if(threadIdx.x == 0 && threadIdx.y == 0)
     {  
        d_a->x=100.0;  
        d_a->y = 100.0;    
     }
}  

int main(void)  
{  
   point * a = (point*)malloc(sizeof(point));  
   a->x=10.0;   
   a->y=10.0;    
   point * d_a;  
   cudaMalloc((void**)&d_a,sizeof(point));  
   cudaMemcpy(d_a,a,sizeof(point),cudaMemcpyHostToDevice);  
   dim3 dimblock(16,16);  
   dim3 dimgrid(1,1);  

   MyFunc<<<dimgrid,dimblock>>>(d_a);  
   cudaMemcpy(a,d_a,sizeof(point),cudaMemcpyDeviceToHost);    
   printf("%lf %lf\n",a->x,a->y);

   return cudaThreadExit();
} 
