#icc propagate-toz-test.C -o propagate-toz-test.exe -fopenmp -O3
CC = icc
LDFLAGS += -fopenmp -O3
CXXFLAGS += -march=native -fopenmp -O3 -DUSE_GPU 
NVCC = nvcc
CUDAFLAGS += -arch=sm_70 -O3 #-rdc=true #-L${CUDALIBDIR} -lcudart 
CUDALDFLAGS += -L${CUDALIBDIR} -lcudart 

propagate : propagate-toz-test-main.o propagate-toz-test.o propagateGPU.o
	$(CC) $(LDFLAGS) $(CUDALDFLAGS) -o propagate propagate-toz-test.o propagate-toz-test-main.o propagateGPU.o
propagate-toz-test.o : propagate-toz-test.C propagate-toz-test.h
	$(CC) $(CXXFLAGS) -o propagate-toz-test.o -c propagate-toz-test.C
propagateGPU.o : propagateGPU.cu propagateGPU.cuh
	$(NVCC) $(CUDAFLAGS) -o propagateGPU.o -c propagateGPU.cu
propagate-toz-test-main.o : propagate-toz-test-main.cc propagate-toz-test.h
	$(CC) $(CXXFLAGS) -o propagate-toz-test-main.o -c propagate-toz-test-main.cc

clean:
	rm -rf propagate *.o 
