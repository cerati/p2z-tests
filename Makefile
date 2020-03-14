########################
# Set the program name #
########################
#BENCHMARK = propagate

###############################################
#    Set macros used for the input program    #
###############################################
# COMPILER options: pgi, gcc, openarc, nvcc   #
# SRCTYPE options: cpp, c ,cu                 #
# MODE options: acc, omp, seq, cuda           #
###############################################
COMPILER ?= nvcc
SRCTYPE ?= cu
MODE ?= cuda

######################################
# Set the input source files (CSRCS) #
######################################
ifeq ($(SRCTYPE),cpp)
ifeq ($(MODE),acc)
CSRCS = propagate-toz-test_OpenACC.cpp
else
CSRCS = propagate-toz-test.cpp
endif
else
ifeq ($(MODE),acc)
CSRCS = propagate-toz-test_OpenACC.c
else ifeq ($(MODE),cuda)
CSRCS = propagate-toz-test_CUDA.cu
else
CSRCS = propagate-toz-test.c
endif
endif


ifeq ($(COMPILER),pgi)
###############
# PGI Setting #
###############
ifeq ($(SRCTYPE),cpp)
CXX=pgc++
ifeq ($(MODE),acc)
CFLAGS1 = -I. -Minfo=acc -fast -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(MODE),omp)
CFLAGS1 = -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(MODE),seq)
CFLAGS1 = -I. -fast 
endif
else
CXX=pgcc
ifeq ($(MODE),acc)
CFLAGS1 = -I. -Minfo=acc -fast -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(MODE),omp)
CFLAGS1 = -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(MODE),seq)
CFLAGS1 = -I. -fast -c99
endif
endif
endif

ifeq ($(COMPILER),gcc)
###############
# GCC Setting #
###############
ifeq ($(SRCTYPE),cpp)
CXX=g++
ifeq ($(MODE),omp)
CFLAGS1 = -O3 -I. -fopenmp 
CLIBS1 = -lm -lgomp
else
CFLAGS1 = -O3 -I. 
CLIBS1 = -lm
endif
else
CXX=gcc
ifeq ($(MODE),omp)
CFLAGS1 = -O3 -I. -std=c99 -fopenmp
CLIBS1 = -lm -lgomp
else
CFLAGS1 = -O3 -I. -std=c99
CLIBS1 = -lm
endif
endif
endif

ifeq ($(COMPILER),openarc)
###################
# OpenARC Setting #
###################
CXX=g++
CSRCS = ./cetus_output/propagate-toz-test_OpenACC.cpp
# On Linux with CUDA GPU
CFLAGS1 = -O3 -I. -I${openarc}/openarcrt 
CLIBS1 = -L${openarc}/openarcrt -lcuda -lopenaccrt_cuda -lomphelper
# On macOS
#CFLAGS1 = -O3 -I. -I${openarc}/openarcrt -arch x86_64
#CLIBS1 = -L${openarc}/openarcrt -lopenaccrt_opencl -lomphelper -framework OpenCL
endif

ifeq ($(COMPILER),intel)
#################
# Intel Setting #
#################
CXX=icc
CFLAGS1= -Wall -I. -O3 -fopenmp -fopenmp-simd
#CFLAGS1= -Wall -I. -O3 -xMIC-AVX512 -qopenmp -qopenmp-offload=host -fimf-precision=low:sqrt,exp,log,/
endif

ifeq ($(COMPILER),ibm)
###############
# IBM Setting #
###############
CXX=xlc
CFLAGS1= -I. -Wall -v -O3 -qsmp=noauto:omp -qnooffload #host power9
endif

ifeq ($(COMPILER),llvm)
################
# LLVM Setting #
################
CXX=clang
CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=x86_64 -lm
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -lm
endif

ifeq ($(COMPILER),nvcc)
################
# NVCC Setting #
################
CXX=nvcc
CFLAGS1= -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread 
CLIBS1= -L${CUDALIBDIR} -lcudart 
endif

################################################
# TARGET is where the output binary is stored. #
################################################
TARGET = ./bin
BENCHMARK = "propagate_$(COMPILER)_$(MODE)"


$(TARGET)/$(BENCHMARK): $(CSRCS)
	if [ ! -d "./bin" ]; then mkdir bin; fi
	$(CXX) $(CFLAGS1) $(CSRCS) $(CLIBS1) -o $(TARGET)/$(BENCHMARK)
	if [ -f "./cetus_output/openarc_kernel.cu" ]; then cp ./cetus_output/openarc_kernel.cu ${TARGET}/; fi
	if [ -f "./cetus_output/openarc_kernel.cl" ]; then cp ./cetus_output/openarc_kernel.cl ${TARGET}/; fi

clean:
	rm -f $(TARGET)/$(BENCHMARK) $(TARGET)/openarc_kernel.* $(TARGET)/*.ptx *.o

purge: clean
	rm -rf bin cetus_output openarcConf.txt 
