########################
# Set the program name #
########################
#BENCHMARK = propagate

##########################################################
#       Set macros used for the input program            #
##########################################################
# COMPILER options: pgi, gcc, openarc, nvcc, icc, llvm   #
#                   ibm                                  #
# MODE options: acc, omp, seq, cuda, tbb, eigen, alpaka, #
#               omp4, cudav2, cudav3, accc, acccv3       #
#               omp4c, omp4cv3, accccpu, acccv3cpu       #
##########################################################
COMPILER ?= nvcc
MODE ?= eigen
OS ?= linux
DEBUG ?= 0
###########Tunable parameters############################
TUNEB ?= 0
TUNETRK ?= 0
TUNEEVT ?= 0
STREAMS ?= 0
NTHREADS ?= 0
NITER ?= 0
NLAYER ?= 0
ifneq ($(TUNEB),0)
TUNE += -Dbsize=$(TUNEB)
endif
ifneq ($(TUNETRK),0)
TUNE += -Dntrks=$(TUNETRK)
endif
ifneq ($(TUNEEVT),0)
TUNE += -Dnevts=$(TUNEEVT)
endif
ifneq ($(STREAMS),0)
TUNE += -Dnum_streams=$(STREAMS)
endif
ifneq ($(NTHREADS),0)
TUNE += -Dnthreads=$(NTHREADS)
endif
ifneq ($(NITER),0)
TUNE += -DNITER=$(NITER)
endif
ifneq ($(NLAYER),0)
TUNE += -Dnlayer=$(NLAYER)
endif
##########CUDA Tunable parameters########################
THREADSX ?= 0
THREADSY ?= 0
BLOCKS ?= 0
ifneq ($(THREADSX),0)
TUNE += -Dthreadsperblockx=$(THREADSX)
endif
ifneq ($(THREADSY),0)
TUNE += -Dthreadsperblocky=$(THREADSY)
endif
ifneq ($(BLOCKS),0)
TUNE += -Dblockspergrid=$(BLOCKS)
endif

################
#  OMP Setting #
################
ifeq ($(MODE),omp)
CSRCS = propagate-toz-test_OMP.cpp
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1 += -O3 -I. -fopenmp 
CLIBS1 += -lm -lgomp
endif
ifeq ($(COMPILER),pgi)
CXX=nvc++
#CXX=pgc++
CFLAGS1 += -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1 += -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high
endif
ifeq ($(COMPILER),llvm)
CXX=clang
CFLAGS1 += -Wall -O3 -I. -fopenmp -fopenmp-targets=x86_64 -lm
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -lm
endif
ifeq ($(COMPILER),ibm)
CXX=xlc
CFLAGS1 += -I. -Wall -v -O3 -qarch=pwr9 -qsmp=noauto:omp -qnooffload #host power9
endif
endif

#################
#  OMP4 Setting #
#################
ifeq ($(MODE),omp4)
CSRCS = propagate-toz-test_OpenMP4_sync_v1.cpp
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1 += -O3 -I. -fopenmp -foffload="-lm"
CLIBS1 += -lm -lgomp 
else ifeq ($(COMPILER),llvm)
CXX=clang++
CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=ppc64le-unknown-linux-gnu
CLIBS1 += -lm
else ifeq ($(COMPILER),ibm)
CXX=xlc++_r
CFLAGS1 += -I. -Wall -v -O3 -qsmp=omp -qoffload #device V100
CLIBS1 += -lm
else ifeq ($(COMPILER),pgi)
CXX=nvc++
CFLAGS1 += -I. -Minfo=mp -O3 -Mfprelaxed -mp=gpu -mcmodel=medium -Mlarge_arrays
CLIBS1 += -lm
else
CSRCS = "NotSupported"
endif
endif

###################
#  OMP4 C Setting #
###################
ifeq ($(MODE),omp4c)
CSRCS = propagate-toz-test_OpenMP4_sync.c
ifeq ($(COMPILER),gcc)
CXX=gcc
CFLAGS1 += -O3 -I. -fopenmp -foffload="-lm"
CLIBS1 += -lm -lgomp 
else ifeq ($(COMPILER),llvm)
CXX=clang
CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=ppc64le-unknown-linux-gnu
CLIBS1 += -lm
else ifeq ($(COMPILER),ibm)
CXX=xlc_r
CFLAGS1 += -I. -Wall -v -O3 -qsmp=omp -qoffload #device V100
CLIBS1 += -lm
else ifeq ($(COMPILER),pgi)
CXX=nvc
CFLAGS1 += -I. -Minfo=mp -O3 -Mfprelaxed -mp=gpu -mcmodel=medium -Mlarge_arrays
CLIBS1 += -lm
else ifeq ($(COMPILER),openarc)
CSRCS = ../cetus_output/propagate-toz-test_OpenMP4_sync.cpp
ifeq ($(OS),linux)
# On Linux with CUDA GPU
ifeq ($(DEBUG),1)
CFLAGS1 += -g -I. -I${openarc}/openarcrt 
else
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt 
endif
else
# On Mac OS
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt -arch x86_64
endif
ifeq ($(OPENARC_ARCH),5)
CXX=hipcc
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_hip -lomphelper
else ifeq ($(OPENARC_ARCH),6)
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lpthread -liris -ldl -lopenaccrt_iris -lomphelper
else ifeq ($(OPENARC_ARCH),1)
ifeq ($(OS),linux)
# On Linux
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lOpenCL -lomphelper
else
# On Mac OS
CXX=clang++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lomphelper -framework OpenCL
endif
else
CXX=g++
ifeq ($(DEBUG),1)
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_cudapf -lcuda -lomphelper
else
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_cuda -lcuda -lomphelper
endif
endif
else
CSRCS = "NotSupported"
endif
endif

###################
#  OMP4 C Setting #
###################
ifeq ($(MODE),omp4cv3)
CSRCS = propagate-toz-test_OpenMP4_async.c
ifeq ($(COMPILER),gcc)
CXX=gcc
CFLAGS1 += -O3 -I. -fopenmp -foffload="-lm"
CLIBS1 += -lm -lgomp 
else ifeq ($(COMPILER),llvm)
CXX=clang
CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=ppc64le-unknown-linux-gnu
CLIBS1 += -lm
else ifeq ($(COMPILER),ibm)
CXX=xlc_r
CFLAGS1 += -I. -Wall -v -O3 -qsmp=omp -qoffload #device V100
CLIBS1 += -lm
else ifeq ($(COMPILER),pgi)
CXX=nvc
CFLAGS1 += -I. -Minfo=mp -O3 -Mfprelaxed -mp=gpu -mcmodel=medium -Mlarge_arrays
CLIBS1 += -lm
else ifeq ($(COMPILER),openarc)
CSRCS = ../cetus_output/propagate-toz-test_OpenMP4_async.cpp
ifeq ($(OS),linux)
# On Linux with CUDA GPU
ifeq ($(DEBUG),1)
CFLAGS1 += -g -I. -I${openarc}/openarcrt 
else
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt 
endif
else
# On Mac OS
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt -arch x86_64
endif
ifeq ($(OPENARC_ARCH),5)
CXX=hipcc
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_hip -lomphelper
else ifeq ($(OPENARC_ARCH),6)
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lpthread -liris -ldl -lopenaccrt_iris -lomphelper
else ifeq ($(OPENARC_ARCH),1)
ifeq ($(OS),linux)
# On Linux
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lOpenCL -lomphelper
else
# On Mac OS
CXX=clang++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lomphelper -framework OpenCL
endif
else
CXX=g++
ifeq ($(DEBUG),1)
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_cudapf -lcuda -lomphelper
else
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_cuda -lcuda -lomphelper
endif
endif
else
CSRCS = "NotSupported"
endif
endif


################
#  ACC Setting #
################
ifeq ($(MODE),acc)
# Synchronous version
#In the following version (V1), PGI incorrectly allocates gang-private data (errorProp, temp, inverse_temp, kGain, newErr) as gang-shared.
#With nvc++ V21.11 with -O3 option, this compiles and run correctly.
#CSRCS = propagate-toz-test_OpenACC_sync_v1.cpp
#In the following version (V2), temporary data (errorProp, temp, inverse_temp, kGain, newErr) are declared as vector-private, 
#which is correct but very inefficient.
#CSRCS = propagate-toz-test_OpenACC_sync_v2.cpp
#In the following version (V3), PGI correctly privatizes the gang-private data (errorProp, temp, inverse_temp, kGain, newErr) in the global memory 
#but not in the shared memory; less optimal.
#CSRCS = propagate-toz-test_OpenACC_sync_v3.cpp
#In the following version (V4), private data (errorProp, temp, inverse_temp, kGain, newErr) are used as vector private with bsize = 1.
#Performs best when using nvc++ V21.11 with -O3 option.
CSRCS = propagate-toz-test_OpenACC_sync_v4.cpp
ifeq ($(COMPILER),pgi)
CXX=nvc++
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays
#CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc -mcmodel=medium -Mlarge_arrays
#CXX=pgc++
#CFLAGS1 += -I. -Minfo=acc -fast -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays
else
CSRCS = "NotSupported"
endif
endif

##################
#  ACC C Setting #
##################
ifeq ($(MODE),accc)
ifeq ($(COMPILER),pgi)
CSRCS = propagate-toz-test_OpenACC_sync.c
CXX=nvc
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays
else ifeq ($(COMPILER),openarc)
#OpenACC C V1: synchronous version
CSRCS = ../cetus_output/propagate-toz-test_OpenACC_sync.cpp
ifeq ($(OS),linux)
# On Linux with CUDA GPU
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt 
else
# On Mac OS
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt -arch x86_64
endif
ifeq ($(OPENARC_ARCH),5)
CXX=hipcc
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_hip -lomphelper
else ifeq ($(OPENARC_ARCH),6)
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lpthread -liris -ldl -lopenaccrt_iris -lomphelper
else ifeq ($(OPENARC_ARCH),1)
ifeq ($(OS),linux)
# On Linux
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lOpenCL -lomphelper
else
# On Mac OS
CXX=clang++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lomphelper -framework OpenCL
endif
else
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_cuda -lcuda -lomphelper
endif
else
CSRCS = "NotSupported"
endif
endif

ifeq ($(MODE),acccv3)
#OpenACC C V3: asynchronous version, which has the same computation/memory mapping as the CUDA V3.
ifeq ($(COMPILER),pgi)
CSRCS = propagate-toz-test_OpenACC_async.c
CXX=nvc
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays
else ifeq ($(COMPILER),openarc)
CSRCS = ../cetus_output/propagate-toz-test_OpenACC_async.cpp
ifeq ($(OS),linux)
# On Linux with CUDA GPU
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt 
else
# On Mac OS
CFLAGS1 += -O3 -I. -I${openarc}/openarcrt -arch x86_64
endif
ifeq ($(OPENARC_ARCH),5)
CXX=hipcc
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_hip -lomphelper
else ifeq ($(OPENARC_ARCH),6)
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lpthread -liris -ldl -lopenaccrt_iris -lomphelper
else ifeq ($(OPENARC_ARCH),1)
ifeq ($(OS),linux)
# On Linux
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lOpenCL -lomphelper
else
# On Mac OS
CXX=clang++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_opencl -lomphelper -framework OpenCL
endif
else
CXX=g++
CLIBS1 += -L${openarc}/openarcrt -lopenaccrt_cuda -lcuda -lomphelper
endif
else
CSRCS = "NotSupported"
endif
endif

##########################
#  ACC C Setting for CPU #
##########################
ifeq ($(MODE),accccpu)
ifeq ($(COMPILER),pgi)
CSRCS = propagate-toz-test_OpenACC_sync.c
CXX=nvc
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc=multicore -mcmodel=medium -Mlarge_arrays
else ifeq ($(COMPILER),openarc)
#OpenACC C V1: synchronous version
CSRCS = ../cetus_output/propagate-toz-test_OpenACC_sync.c
CXX=gcc
CFLAGS1 += -O3 -I. -fopenmp 
CLIBS1 += -lm -lgomp
else
CSRCS = "NotSupported"
endif
endif

##########################
#  ACC C Setting for CPU #
##########################
ifeq ($(MODE),acccv3cpu)
ifeq ($(COMPILER),pgi)
CSRCS = propagate-toz-test_OpenACC_async.c
CXX=nvc
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc=multicore -mcmodel=medium -Mlarge_arrays
else ifeq ($(COMPILER),openarc)
#OpenACC C V3: asynchronous version, which has the same computation/memory mapping as the CUDA V3.
CSRCS = ../cetus_output/propagate-toz-test_OpenACC_async.c
CXX=gcc
CFLAGS1 += -O3 -I. -fopenmp 
CLIBS1 += -lm -lgomp
else
CSRCS = "NotSupported"
endif
endif

################
#  TBB Setting #
################
ifeq ($(MODE),tbb)
CSRCS = propagate-toz-test_tbb.cpp
TBB_PREIX := /opt/intel
CLIBS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1+=  -fopenmp -O3 -I.
CLIBS1 += -lm -lgomp -L/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1+= -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high
endif
ifeq ($(COMPILER),pgi)
CXX=nvc++
#CXX=pgc++
CFLAGS1 += -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
endif


#################
#  CUDA Setting #
#################
ifeq ($(MODE),cuda)
CSRCS = propagate-toz-test_CUDA.cu
#CSRCS = propagate-toz-test_CUDA_v2.cu
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CFLAGS1 += -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread
CLIBS1 += -L${CUDALIBDIR} -lcudart 
endif
endif

ifeq ($(MODE),cudav2)
#CSRCS = propagate-toz-test_CUDA.cu
CSRCS = propagate-toz-test_CUDA_v2.cu
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CFLAGS1 += -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread -maxrregcount 64
CLIBS1 += -L${CUDALIBDIR} -lcudart 
endif
endif

ifeq ($(MODE),cudav3)
CSRCS = propagate-toz-test_CUDA_v3.cu
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CFLAGS1 += -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread -maxrregcount 64
CLIBS1 += -L${CUDALIBDIR} -lcudart 
endif
endif

##################
#  EIGEN Setting #
##################
ifeq ($(MODE),eigen)
CSRCS = propagate-toz-test_Eigen.cpp
CLIBS1 += -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub -L${CUDALIBDIR} -lcudart 
#CLIBS1 += -Ieigen -I/mnt/data1/dsr/cub -L${CUDALIBDIR} -lcudart 
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1 += -fopenmp -O3 -fopenmp-simd -lm -lgomp -march=native 
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1 += -fopenmp -O3 -fopenmp-simd  -mtune=native -march=native -xHost -qopt-zmm-usage=high
endif
ifeq ($(COMPILER),pgi)
CXX=nvc++
#CXX=pgc++
CFLAGS1 += -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CSRCS = propagate-toz-test_Eigen.cu
CFLAGS1 += -arch=sm_70 --default-stream per-thread -O3 --expt-relaxed-constexpr -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub
#CFLAGS1 += -arch=sm_70 --default-stream per-thread -O3 --expt-relaxed-constexpr -Ieigen -I/mnt/data1/dsr/cub
endif
endif

###################
#  ALPAKA Setting #
###################
ifeq ($(MODE),alpaka)
CSRCS = propagate-toz-test_alpaka.cpp
TBB_PREIX := /opt/intel
CLIBS1 = -I/mnt/data1/mgr85/p2z-tests/alpaka_lib/include 
CLIBS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1+= -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
CFLAGS1+= -fopenmp -O3 -I.
#CLIBS1 += -lm -lgomp 
CLIBS1 += -lm -lgomp -L/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8
endif
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CSRCS = propagate-toz-test_alpaka.cu
CFLAGS1+= -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread -DALPAKA_ACC_GPU_CUDA_ENABLED --expt-relaxed-constexpr --expt-extended-lambda 
CFLAGS1+= -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#CFLAGS1+= -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
CLIBS1 += -L${CUDALIBDIR} -lcudart -g
endif
endif


################################################
# TARGET is where the output binary is stored. #
################################################
TARGET = ./bin
BENCHMARK = "propagate_$(COMPILER)_$(MODE)"

ifneq ($(MODE),omp4c)
ifneq ($(MODE),omp4cv3)
ifneq ($(MODE),accc)
ifneq ($(MODE),acccv3)
ifneq ($(MODE),accccpu)
ifneq ($(MODE),acccv3cpu)
ifeq ($(COMPILER),ibm)
CFLAGS1 += -std=c++11
else
CFLAGS1 += -std=c++17
endif
endif
endif
endif
endif
endif
endif

$(TARGET)/$(BENCHMARK): precmd src/$(CSRCS)
	if [ ! -d "$(TARGET)" ]; then mkdir bin; fi
	$(CXX) $(CFLAGS1) src/$(CSRCS) $(CLIBS1) $(TUNE) -o $(TARGET)/$(BENCHMARK)
	if [ -f $(TARGET)/*.ptx ]; then rm $(TARGET)/*.ptx; fi
	if [ -f "./cetus_output/openarc_kernel.cu" ]; then cp ./cetus_output/openarc_kernel.cu ${TARGET}/; fi
	if [ -f "./cetus_output/openarc_kernel.cl" ]; then cp ./cetus_output/openarc_kernel.cl ${TARGET}/; fi
	if [ -f "./cetus_output/openarc_kernel.hip.cpp" ]; then cp ./cetus_output/openarc_kernel.hip.cpp ${TARGET}/; fi

precmd:
	@if [ "$(CSRCS)" = "NotSupported" ]; then echo "==> [ERROR] The compiler $(COMPILER) is not supported for the mode = $(MODE)"; fi
	@if [ "$(COMPILER)" = "openarc" ]; then rm -rf ./cetus_output $(TARGET)/openarc_kernel.* $(TARGET)/*.ptx $(TARGET)/propagate_openarc_*; fi
	@if [ "$(COMPILER)" = "openarc" ]; then O2GBuild.script $(MODE); fi

clean:
	rm -f $(TARGET)/$(BENCHMARK) $(TARGET)/openarc_kernel.* $(TARGET)/*.ptx *.o

cleanall: purge
	rm -f $(TARGET)/* 

purge: clean
	rm -rf cetus_output openarcConf.txt 
