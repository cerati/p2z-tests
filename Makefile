#icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
CC = icc #gcc throws errors for sinf functions etc 
PGC = pgc++ -acc -L${PGI} -ta=tesla:managed -fPIC -Minfo -Mfprelaxed 
#LDFLAGS += -fopenmp -O3
CXXFLAGS +=  -fopenmp -O3  #-DUSE_GPU 
NVCC = nvcc
CUDAFLAGS += -arch=sm_70 -O3 -g -G#-rdc=true #-L${CUDALIBDIR} -lcudart 
CUDALDFLAGS += -L${CUDALIBDIR} -lcudart

#TYPE = icc
TYPE = cuda
#TYPE = pgi


ifeq ($(TYPE),icc)
COMP = ${CC}
FLAGS = ${CXXFLAGS}
endif
ifeq ($(TYPE),cuda)
COMP = ${CC}
FLAGS = ${CXXFLAGS} -DUSE_GPU #--default-stream per-thread #${CXXFLAGS} -DUSE_GPU 
endif


ifeq ($(TYPE),pgi)
COMP = ${PGC}
FLAGS = -DUSE_ACC
endif



propagate : propagate-toz-test-main.o propagate-toz-test.o propagateGPU.o
	$(COMP) $(FLAGS) $(CUDALDFLAGS) -o propagate propagate-toz-test.o propagate-toz-test-main.o propagateGPU.o
propagate-toz-test.o : propagate-toz-test.C propagate-toz-test.h
	$(COMP) $(FLAGS) -o propagate-toz-test.o -c propagate-toz-test.C
propagateGPU.o : propagateGPU.cu propagateGPUStructs.cuh propagateGPU.cuh
	$(NVCC) --default-stream per-thread $(CUDAFLAGS) -o propagateGPU.o -c propagateGPU.cu
propagate-toz-test-main.o : propagate-toz-test-main.cc propagate-toz-test.h
	$(COMP) $(FLAGS) -o propagate-toz-test-main.o -c propagate-toz-test-main.cc

clean:
	rm -rf propagate *.o 
