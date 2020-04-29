########################
# Set the program name #
########################
#BENCHMARK = propagate

##########################################################
#       Set macros used for the input program            #
##########################################################
# COMPILER options: pgi, gcc, openarc, nvcc, icc         #
# SRCTYPE options: cpp, c ,cu                            #
# MODE options: acc, omp, seq, cuda, tbb, eigen, alpaka  #
##########################################################
COMPILER ?= openarc
SRCTYPE ?= cpp
MODE ?= acc

######################################
# Set the input source files (CSRCS) #
######################################
ifeq ($(SRCTYPE),cpp)
ifeq ($(MODE),acc)
CSRCS = propagate-toz-test_OpenACC.cpp
else
CSRCS = propagate-toz-test_v3.cpp
endif
else
ifeq ($(MODE),acc)
CSRCS = propagate-toz-test_OpenACC.c
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
ifeq ($(MODE),eigen)
CSRCS = propagate-toz-test_Eigen.cpp
CFLAGS1= -fopenmp -O3 -fopenmp-simd -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub -lm -lgomp 
else
ifeq ($(MODE),tbb)
CSRCS = propagate-toz-test_tbb.cpp
TBB_PREIX := /opt/intel
CFLAGS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb
CFLAGS1+= -L/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8 -fopenmp -O3 -I.
CLIBS1 = -lm -lgomp
else
ifeq ($(MODE),alpaka)
CSRCS = propagate-toz-test_alpaka.cpp
#CSRCS = alpaka_test.cpp
CFLAGS1+= -I/mnt/data1/mgr85/p2z-tests/alpaka_lib/include -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
TBB_PREIX := /opt/intel
CFLAGS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
CFLAGS1+= -fopenmp -O3 -I.
CLIBS1 += -lm -lgomp
else
CFLAGS1 = -O3 -I. 
CLIBS1 = -lm
endif
endif
endif
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

ifeq ($(COMPILER),icc)
#################
# Intel Setting #
#################
CXX=icc
ifeq ($(MODE),tbb)
CSRCS = propagate-toz-test_tbb.cpp
TBB_PREIX := /opt/local
CFLAGS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb
CFLAGS1+= -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high
else
ifeq ($(MODE),eigen)
CSRCS = propagate-toz-test_Eigen.cpp
CXX=icc
CFLAGS1= -fopenmp -O3 -fopenmp-simd -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub -mtune=native -march=native -xHost -qopt-zmm-usage=high
else
CFLAGS1= -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high
#CFLAGS1= -Wall -I. -O3 -xMIC-AVX512 -qopenmp -qopenmp-offload=host -fimf-precision=low:sqrt,exp,log,/
endif
endif
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
ifeq ($(MODE),eigen)
CSRCS = propagate-toz-test_Eigen.cu
CXX=nvcc
CFLAGS1= -arch=sm_70 --default-stream per-thread -O3 --expt-relaxed-constexpr -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub
CLIBS1= -L${CUDALIBDIR} -lcudart 
else
ifeq ($(MODE),alpaka)
CXX=nvcc
CSRCS = propagate-toz-test_alpaka.cu
#CSRCS = alpaka_test.cpp
#CFLAGS1+= -I/mnt/data1/mgr85/p2z-tests/include -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
CFLAGS1+= -I/mnt/data1/mgr85/p2z-tests/alpaka_lib/include -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread -L${CUDALIBDIR} -lcudart -DALPAKA_ACC_GPU_CUDA_ENABLED --expt-relaxed-constexpr --expt-extended-lambda 
#CFLAGS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
else
CSRCS = propagate-toz-test_CUDA.cu
CXX=nvcc
CFLAGS1= -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread 
CLIBS1= -L${CUDALIBDIR} -lcudart 
endif
endif
endif

################################################
# TARGET is where the output binary is stored. #
################################################
TARGET = ./bin
BENCHMARK = "propagate_$(COMPILER)_$(MODE)"


$(TARGET)/$(BENCHMARK): $(CSRCS)
	if [ ! -d "$(TARGET)" ]; then mkdir bin; fi
	$(CXX) $(CFLAGS1) $(CSRCS) $(CLIBS1) -o $(TARGET)/$(BENCHMARK)
	if [ -f $(TARGET)/*.ptx ]; then rm $(TARGET)/*.ptx; fi
	if [ -f "./cetus_output/openarc_kernel.cu" ]; then cp ./cetus_output/openarc_kernel.cu ${TARGET}/; fi
	if [ -f "./cetus_output/openarc_kernel.cl" ]; then cp ./cetus_output/openarc_kernel.cl ${TARGET}/; fi

clean:
	rm -f $(TARGET)/$(BENCHMARK) $(TARGET)/openarc_kernel.* $(TARGET)/*.ptx *.o

purge: clean
	rm -rf bin cetus_output openarcConf.txt 
