#icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
CC = icc #gcc throws errors for sinf functions etc 
PGC = pgc++ -acc -L${PGI} -ta=tesla:managed -fPIC -Minfo -Mfprelaxed 
#LDFLAGS += -fopenmp -O3
CXXFLAGS +=  -fopenmp -O3  #-DUSE_GPU 
NVCC = nvcc
CUDAFLAGS += -arch=sm_70 -O3 -DUSE_GPU#-rdc=true #-L${CUDALIBDIR} -lcudart 
CUDALDFLAGS += -L${CUDALIBDIR} -lcudart

#TYPE = icc
TYPE = cuda
#TYPE = pgi


ifeq ($(TYPE),icc)
COMP = ${CC}
FLAGS = ${CXXFLAGS}
endif
ifeq ($(TYPE),cuda)
COMP = ${NVCC}
FLAGS =  ${CUDAFLAGS} --default-stream per-thread #${CXXFLAGS} -DUSE_GPU 
endif


ifeq ($(TYPE),pgi)
COMP = ${PGC}
FLAGS = -DUSE_ACC
endif


#propagate : propagate-toz-test-main.o propagate-toz-test.o propagateGPU.o
#	$(COMP) $(FLAGS) $(CUDALDFLAGS) -o propagate propagate-toz-test.o propagate-toz-test-main.o propagateGPU.o
#propagate-toz-test.o : propagate-toz-test.C propagate-toz-test.h propagateGPU.cuh
#	$(COMP) $(FLAGS) -o propagate-toz-test.o -c propagate-toz-test.C
#propagateGPU.o : propagateGPU.cu propagateGPU.cuh
#	$(NVCC) --default-stream per-thread $(CUDAFLAGS) -o propagateGPU.o -c propagateGPU.cu
#propagate-toz-test-main.o : propagate-toz-test-main.cc propagate-toz-test.h
#	$(COMP) $(FLAGS) -o propagate-toz-test-main.o -c propagate-toz-test-main.cc

ifeq ($(TYPE),cuda)
propagate : propagate-toz-test-main.o propagate-toz-test.o propagateGPU.o
	$(COMP) $(FLAGS) $(CUDALDFLAGS) -o propagate propagate-toz-test-main.o propagateGPU.o
propagateGPU.o : propagateGPU.cu propagateGPU.h
	$(NVCC) $(FLAGS) -o propagateGPU.o -c propagateGPU.cu
propagate-toz-test-main.o : propagate-toz-test-main.cc propagateGPU.h
	$(COMP) $(FLAGS) -o propagate-toz-test-main.o -c propagate-toz-test-main.cc

else
#propagate : propagate-toz-test-main.o propagate-toz-test.o propagateGPU.o
#	$(COMP) $(FLAGS) $(CUDALDFLAGS) -o propagate propagate-toz-test.o propagate-toz-test-main.o propagateGPU.o
propagate : propagate-toz-test-main.o propagate-toz-test.o
	$(COMP) $(FLAGS) $(CUDALDFLAGS) -o propagate propagate-toz-test.o propagate-toz-test-main.o
propagate-toz-test.o : propagate-toz-test.C propagate-toz-test.h
#propagate-toz-test.o : propagate-toz-test.C propagate-toz-test.h propagateGPU.cuh
	$(COMP) $(FLAGS) -o propagate-toz-test.o -c propagate-toz-test.C
#propagateGPU.o : propagateGPU.cu propagateGPU.cuh
#	$(NVCC) --default-stream per-thread $(CUDAFLAGS) -o propagateGPU.o -c propagateGPU.cu
propagate-toz-test-main.o : propagate-toz-test-main.cc propagate-toz-test.h
	$(COMP) $(FLAGS) -o propagate-toz-test-main.o -c propagate-toz-test-main.cc
endif
clean:
	rm -rf propagate *.o 
