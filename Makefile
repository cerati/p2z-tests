########################
# Set the program name #
########################
#BENCHMARK = propagate

##########################################################
#       Set macros used for the input program            #
##########################################################
# COMPILER options: gcc, openarc, nvcc, icc, llvm        #
#                   ibm, nvhpc, dpcpp                    #
# MODE options: omp, seq, tbb, eigen, alpaka, alpakav4,  #
#               cuda, cudav1, cudav2, cudav3, cudav4,    #
#               cudauvm, cudahyb, pstl, accc, acccv3,    #
#               acccv4, omp4c, omp4cv3, omp4cv4, accccpu,#
#               acccv3cpu, acccv4cpu, kokkosv1, kokkosv2,#
#               kokkosv3, kokkosv4, kokkosv5, kokkosv6   #
##########################################################
COMPILER ?= nvcc
MODE ?= eigen
OS ?= linux
DEBUG ?= 0
CUDA_ARCH ?= 70
#GCC_ROOT ?= /sw/summit/gcc/11.1.0-2
GCC_ROOT ?= g++
INCLUDE_DATA ?= 1
ifneq ($(INCLUDE_DATA),0)
TUNE += -Dinclude_data=$(INCLUDE_DATA)
endif
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
else
ifeq ($(INCLUDE_DATA),0)
TUNE += -DNITER=100
NITER = 100
else
TUNE += -DNITER=10
NITER = 10
endif
endif
ifneq ($(NLAYER),0)
TUNE += -Dnlayer=$(NLAYER)
endif
##########CUDA Tunable parameters########################
THREADSX ?= 0
THREADSY ?= 0
BLOCKS ?= 0
USE_ASYNC ?= 1
USE_FMAD ?= 1
USE_GPU ?= 1
ifneq ($(THREADSX),0)
TUNE += -Dthreadsperblockx=$(THREADSX)
endif
ifneq ($(THREADSY),0)
TUNE += -Dthreadsperblocky=$(THREADSY)
endif
ifneq ($(BLOCKS),0)
TUNE += -Dblockspergrid=$(BLOCKS)
endif
ifneq ($(USE_ASYNC),0)
TUNE += -DUSE_ASYNC
endif

##########KOKKOS Make Options########################
# Assume that KOKKOS_ROOT is set to the Kokkos root directory
KOKKOS_PATH ?= $(KOKKOS_ROOT)
KOKKOS_DEVICES ?= Cuda
KOKKOS_ARCH ?= None
PREPIN_HOSTMEM ?= 1

##########ALPAKA Make Options########################
# Assume that ALPAKA_INSTALL_ROOT is set to the Alpaka install root directory
ALPAKA_PATH ?= $(ALPAKA_INSTALL_ROOT)
# Set ALPAKASRC to alpaka_src_gpu to use versions in alpata_src_gpu directory.
ALPAKASRC ?= alpaka_src_gpu

################
#  OMP Setting #
################
ifeq ($(MODE),omp)
CSRCS = propagate-toz-test_OMP.cpp
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1 += -O3 -I. -fopenmp -march=native -mprefer-vector-width=512
CLIBS1 += -lm -lgomp
endif
ifeq ($(COMPILER),nvhpc)
CXX=nvc++
CFLAGS1 += -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1 += -Wall -I. -O3 -fopenmp -xHost -qopt-zmm-usage=high
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

################
#  PSTL Setting #
################
ifeq ($(MODE),pstl)
CSRCS = propagate-toz-test_pstl_v2.cpp
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1 += -O3 -I. -fopenmp 
CLIBS1 += -lm -lgomp -L/opt/intel/tbb-gnu9.3/lib -ltbb
endif
ifeq ($(COMPILER),nvhpc)
CXX=nvc++
ifeq ($(USE_FMAD),1)
CFLAGS1 += -O3 -stdpar -gpu=cc$(CUDA_ARCH) --gcc-toolchain=$(GCC_ROOT) -Mfma
else
CFLAGS1 += -O3 -stdpar -gpu=cc$(CUDA_ARCH) --gcc-toolchain=$(GCC_ROOT) -Mnofma
endif
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1 += -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high
endif
ifeq ($(COMPILER),llvm)
CSRCS = propagate-toz-test_pstl_dpcpp.cpp	
CXX=dpcpp
CFLAGS1 += -O2
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -lm
endif
ifeq ($(COMPILER),ibm)
CXX=xlc
CFLAGS1 += -I. -Wall -v -O3 -qarch=pwr9 -qsmp=noauto:omp -qnooffload #host power9
endif
endif


###################
#  OMP4 C Setting #
###################
COMPILE_OMP4_DEVICE=0

ifeq ($(MODE),omp4c)
CSRCSBASE = propagate-toz-test_OpenMP4_sync
COMPILE_OMP4_DEVICE=1
endif
ifeq ($(MODE),omp4cv3)
CSRCSBASE = propagate-toz-test_OpenMP4_async
COMPILE_OMP4_DEVICE=1
endif
ifeq ($(MODE),omp4cv4)
CSRCSBASE = propagate-toz-test_OpenMP4_async_v4
COMPILE_OMP4_DEVICE=1
endif

ifeq ($(COMPILE_OMP4_DEVICE),1)
CSRCS = $(CSRCSBASE).c
ifeq ($(COMPILER),gcc)
CXX=gcc
CFLAGS1 += -O3 -I. -fopenmp -foffload="-lm -O3"
CLIBS1 += -lm -lgomp 
else ifeq ($(COMPILER),llvm)
CXX=clang
CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_$(CUDA_ARCH)
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
#CFLAGS1 = -Wall -O3 -I. -fopenmp -fopenmp-targets=ppc64le-unknown-linux-gnu
CLIBS1 += -lm
else ifeq ($(COMPILER),ibm)
CXX=xlc_r
CFLAGS1 += -I. -Wall -v -O3 -qsmp=omp -qoffload #device V100
CLIBS1 += -lm
else ifeq ($(COMPILER),nvhpc)
CXX=nvc
CFLAGS1 += -I. -Minfo=mp -O3 -Mfprelaxed -mp=gpu -mcmodel=medium -Mlarge_arrays
CLIBS1 += -lm
else ifeq ($(COMPILER),openarc)
CSRCS = ../cetus_output/$(CSRCSBASE).cpp
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


##################
#  ACC C Setting #
##################
COMPILE_ACC_DEVICE=0

ifeq ($(MODE),accc)
#OpenACC C V1: synchronous version
CSRCSBASE = propagate-toz-test_OpenACC_sync
COMPILE_ACC_DEVICE=1
endif
ifeq ($(MODE),acccv3)
#OpenACC C V3: asynchronous version, which has the same computation/memory mapping as the CUDA V3.
CSRCSBASE = propagate-toz-test_OpenACC_async
COMPILE_ACC_DEVICE=1
endif
ifeq ($(MODE),acccv4)
#OpenACC C V4: asynchronous version, which has the same computation/memory mapping as the CUDA V4.
CSRCSBASE = propagate-toz-test_OpenACC_async_v4
COMPILE_ACC_DEVICE=1
endif

ifeq ($(COMPILE_ACC_DEVICE),1)
CSRCS = $(CSRCSBASE).c
ifeq ($(COMPILER),nvhpc)
CXX=nvc
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc -mcmodel=medium -Mlarge_arrays
ifeq ($(USE_FMAD),1)
CFLAGS1 += -gpu=cc$(CUDA_ARCH) -O3 -Mfma
else
CFLAGS1 += -gpu=cc$(CUDA_ARCH) -O3 -Mnofma
endif
else ifeq ($(COMPILER),gcc)
CXX=gcc
CFLAGS1 += -O3 -I. -fopenacc -foffload="-lm -O3"
CLIBS1 += -lm
else ifeq ($(COMPILER),openarc)
CSRCS = ../cetus_output/$(CSRCSBASE).cpp
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
COMPILE_ACC_HOST=0
ifeq ($(MODE),accccpu)
#OpenACC C V1: synchronous version
CSRCSBASE = propagate-toz-test_OpenACC_sync
COMPILE_ACC_HOST=1
endif
ifeq ($(MODE),acccv3cpu)
#OpenACC C V3: asynchronous version, which has the same computation/memory mapping as the CUDA V3.
CSRCSBASE = propagate-toz-test_OpenACC_async
COMPILE_ACC_HOST=1
endif
ifeq ($(MODE),acccv4cpu)
#OpenACC C V4: asynchronous version, which has the same computation/memory mapping as the CUDA V4.
CSRCSBASE = propagate-toz-test_OpenACC_async_v4
COMPILE_ACC_HOST=1
endif

ifeq ($(COMPILE_ACC_HOST),1)
CSRCS = $(CSRCSBASE).c
ifeq ($(COMPILER),nvhpc)
CXX=nvc
CFLAGS1 += -I. -Minfo=acc -O3 -Mfprelaxed -acc=multicore -mcmodel=medium -Mlarge_arrays
else ifeq ($(COMPILER),gcc)
CXX=gcc
CFLAGS1 += -O3 -I. -fopenacc -foffload=disable -foffload="-lm -O3"
CLIBS1 += -lm
else ifeq ($(COMPILER),openarc)
CSRCS = ../cetus_output/$(CSRCSBASE).c
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
TBB_PREFIX := /packages/intel/oneapi/tbb/2021.1.1
#TBB_PREFIX := /packages/intel/oneapi/tbb/2021.1.1 # /opt/intel/tbb-gnu9.3
CLIBS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib/intel64/gcc4.8 -Wl,-rpath,${TBB_PREFIX}/lib -ltbb
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1+=  -fopenmp -O3 -I. -march=native -mprefer-vector-width=512
TBB_PREFIX := /packages/intel/oneapi/tbb/2021.1.1
CLIBS1 += -lm -lgomp -L${TBB_PREFIX}/lib/intel64/gcc4.8 -ltbb
#-L/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1+= -Wall -I. -O3 -fopenmp -xHost -qopt-zmm-usage=high
endif
ifeq ($(COMPILER),nvhpc)
CXX=nvc++
CFLAGS1 += -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
endif


#################
#  CUDA Setting #
#################
COMPILE_CUDA = 0
ifeq ($(MODE),cuda)
# CUDA_v0 use USM but has the same computation patterns and communication patterns as CUDA_v3
CSRCS = propagate-toz-test_CUDA_v0.cu
COMPILE_CUDA = 1
endif

ifeq ($(MODE),cudav1)
# CUDA_v1 use USM but has the same computation patterns and communication patterns as CUDA_v4
CSRCS = propagate-toz-test_CUDA_v1.cu
COMPILE_CUDA = 1
endif

ifeq ($(MODE),cudav2)
CSRCS = propagate-toz-test_CUDA_v2.cu
COMPILE_CUDA = 1
endif

ifeq ($(MODE),cudav3)
CSRCS = propagate-toz-test_CUDA_v3.cu
COMPILE_CUDA = 1
endif

ifeq ($(MODE),cudav4)
CSRCS = propagate-toz-test_CUDA_v4.cu
COMPILE_CUDA = 1
endif

ifeq ($(MODE),cudauvm)
CSRCS = propagate-toz-test_cuda_uvm_v2.cu
COMPILE_CUDA = 1
endif

ifeq ($(COMPILE_CUDA),1)
ifeq ($(COMPILER),nvcc)
CXX=nvcc
ifeq ($(USE_FMAD),1)
# fmad, which is enabled by default, makes different CUDA versions generate different outputs.
CFLAGS1 += -arch=sm_$(CUDA_ARCH) -O3 -DUSE_GPU --default-stream per-thread -maxrregcount 64 --expt-relaxed-constexpr 
else
CFLAGS1 += -arch=sm_$(CUDA_ARCH) -O3 -DUSE_GPU --default-stream per-thread -maxrregcount 64 --expt-relaxed-constexpr --fmad false
endif
CLIBS1 += -L${CUDALIBDIR} -lcudart 
endif
ifeq ($(COMPILER),nvhpc)
CXX=nvc++
ifeq ($(USE_FMAD),1)
CFLAGS1 += -gpu=cc$(CUDA_ARCH) -O3 -Mfma
else
CFLAGS1 += -gpu=cc$(CUDA_ARCH) -O3 -Mnofma
endif
CLIBS1 += -lcudart 
endif
endif

ifeq ($(MODE),cudahyb)
CSRCS = propagate-toz-test_cuda_hybrid.cpp
ifeq ($(COMPILER),nvhpc)
CXX=nvc++
ifeq ($(USE_FMAD),1)
CFLAGS1 += -cuda -stdpar=gpu -gpu=cc$(CUDA_ARCH) -O3 --gcc-toolchain=$(GCC_ROOT) -gpu=managed -Mfma
else
CFLAGS1 += -cuda -stdpar=gpu -gpu=cc$(CUDA_ARCH) -O3 --gcc-toolchain=$(GCC_ROOT) -gpu=managed -Mnofma
endif
CLIBS1 += -lcudart 
endif
endif

##################
#  EIGEN Setting #
##################
ifeq ($(MODE),eigen)
CSRCS = propagate-toz-test_Eigen.cpp
EIGEN_ROOT ?= /mnt/data1/dsr/mkfit-hackathon/eigen
#CLIBS1 += -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub -L${CUDALIBDIR} -lcudart 
#CLIBS1 += -Ieigen -I/mnt/data1/dsr/cub -L${CUDALIBDIR} -lcudart 
CLIBS1 += -I${EIGEN_ROOT} -lcudart
ifeq ($(COMPILER),gcc)
CXX=g++
CFLAGS1 += -fopenmp -O3 -fopenmp-simd -lm -lgomp -march=native
endif
ifeq ($(COMPILER),icc)
CXX=icc
CFLAGS1 += -fopenmp -O3 -fopenmp-simd  -mtune=native -march=native -xHost -qopt-zmm-usage=high
endif
ifeq ($(COMPILER),nvhpc)
CXX=nvc++
CFLAGS1 += -I. -Minfo=mp -fast -mp -Mnouniform -mcmodel=medium -Mlarge_arrays
endif
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CSRCS = propagate-toz-test_Eigen.cu
#CFLAGS1 += -arch=sm_$(CUDA_ARCH) --default-stream per-thread -O3 --expt-relaxed-constexpr -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub
#CFLAGS1 += -arch=sm_$(CUDA_ARCH) --default-stream per-thread -O3 --expt-relaxed-constexpr -Ieigen -I/mnt/data1/dsr/cub
CFLAGS1 += -arch=sm_$(CUDA_ARCH) --default-stream per-thread -O3 --expt-relaxed-constexpr -I${EIGEN_ROOT}
endif
endif

###################
#  ALPAKA Setting #
###################
COMPILE_CMAKE = 0
CMAKECMD = pwd #dummy command 
COMPILE_ALPAKA = 0

ifeq ($(ALPAKASRC),alpaka_src_gpu)
########################################################################################
# New commands to compile src/alpaka_src_gpu/src/propagate-toz-test_alpaka_cpu_gpu.cpp #
########################################################################################
ifeq ($(MODE),alpaka)
CSRCSDIR = alpaka_src_gpu/src
CSRCS = ${CSRCSDIR}/propagate-toz-test_alpaka_cpu_gpu.cpp
COMPILE_ALPAKA = 1
endif
ifeq ($(MODE),alpakav4)
CSRCSDIR = alpaka_src_gpu/src
CSRCS = ${CSRCSDIR}/propagate-toz-test_alpaka_cpu_gpu_v4.cpp
COMPILE_ALPAKA = 1
endif

ifneq ($(COMPILE_ALPAKA),0)
COMPILE_CMAKE = 1
CMAKE_FLAGS += -DMODE=$(MODE)
CMAKEDIR = alpaka_src_gpu
CMAKEOUTPUT = alpaka_gpu_src
ifneq ($(NITER),0)
CMAKE_FLAGS += -DNITER=$(NITER)
else
ifeq ($(INCLUDE_DATA),0)
CMAKE_FLAGS += -DNITER=100
else
CMAKE_FLAGS += -DNITER=10
endif
endif
ifneq ($(NLAYER),0)
CMAKE_FLAGS += -DNLAYER=$(NLAYER)
endif
ifneq ($(INCLUDE_DATA),0)
CMAKE_FLAGS += -DINCLUDE_DATA=$(INCLUDE_DATA)
endif
CMAKE_FLAGS += -DUSE_FMAD=$(USE_FMAD)
ifneq ($(STREAMS),0)
CMAKE_FLAGS += -Dnum_streams=$(STREAMS)
endif
ifeq ($(COMPILER),gcc)
#For serial CPU
#CMAKECMD = cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=$(ALPAKA_INSTALL_ROOT) -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=on -DALPAKA_CXX_STANDARD=17 -DDEVICE_TYPE=2 $(CMAKE_FLAGS) ..
#For OpenMP CPU
CMAKECMD = cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=$(ALPAKA_INSTALL_ROOT) -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=on -DALPAKA_CXX_STANDARD=17 -DDEVICE_TYPE=3 $(CMAKE_FLAGS) ..
endif
ifeq ($(COMPILER),nvcc)
CMAKECMD = cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=$(ALPAKA_INSTALL_ROOT) -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_CUDA_COMPILER=nvcc -DALPAKA_CXX_STANDARD=17 -DALPAKA_ACC_GPU_CUDA_ENABLE=on -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) -DDEVICE_TYPE=1 $(CMAKE_FLAGS) ..
endif
endif

else
###########################################################
# Old commands to compile src/propagate-toz-test_alpaka.* #
###########################################################
ifeq ($(MODE),alpaka)
CLIBS1 = -I${ALPAKA_INSTALL_ROOT}/include 
ifeq ($(COMPILER),gcc)
CSRCS = propagate-toz-test_alpaka.cpp
TBB_PREIX := /opt/intel
CLIBS1+= -I${TBB_PREFIX}/include -L${TBB_PREFIX}/lib -Wl,-rpath,${TBB_PREFIX}/lib -ltbb
CXX=g++
CFLAGS1+= -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
CFLAGS1+= -fopenmp -O3 -I.
#CLIBS1 += -lm -lgomp 
CLIBS1 += -lm -lgomp -L/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8
endif
ifeq ($(COMPILER),nvcc)
CXX=nvcc
CSRCS = propagate-toz-test_alpaka.cu
CFLAGS1+= -arch=sm_$(CUDA_ARCH) -O3 --default-stream per-thread -DALPAKA_ACC_GPU_CUDA_ENABLED --expt-relaxed-constexpr --expt-extended-lambda
CFLAGS1+= -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#CFLAGS1+= -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
CLIBS1 += -L${CUDALIBDIR} -lcudart -g
endif
endif
endif

###################
#  Kokkos Setting #
###################
COMPILE_KOKKOS = 0

ifeq ($(MODE),kokkosv1)
CSRCSDIR = kokkos_src_v1
CSRCS = $(CSRCSDIR)/propagate-toz-test_Kokkos_v1.cpp
BENCHMARK = propagate_$(COMPILER)_$(MODE)
COMPILE_KOKKOS = 1
endif
ifeq ($(MODE),kokkosv2)
CSRCSDIR = kokkos_src_v2
CSRCS = $(CSRCSDIR)/propagate-toz-test_Kokkos_v2.cpp
BENCHMARK = propagate_$(COMPILER)_$(MODE)
COMPILE_KOKKOS = 1
endif
ifeq ($(MODE),kokkosv3)
CSRCSDIR = kokkos_src_v3
CSRCS = $(CSRCSDIR)/propagate-toz-test_Kokkos_v3.cpp
BENCHMARK = propagate_$(COMPILER)_$(MODE)
COMPILE_KOKKOS = 1
endif
ifeq ($(MODE),kokkosv4)
CSRCSDIR = kokkos_src_v4
CSRCS = $(CSRCSDIR)/propagate-toz-test_Kokkos_v4.cpp
ifneq ($(PREPIN_HOSTMEM),1)
KOKKOS_FLAGS += prepin_hostmem=$(PREPIN_HOSTMEM)
BENCHMARK = propagate_$(COMPILER)_$(MODE)
else
BENCHMARK = propagate_$(COMPILER)_$(MODE)_prepin_host
endif
COMPILE_KOKKOS = 1
endif
ifeq ($(MODE),kokkosv5)
CSRCSDIR = kokkos_src_v5
CSRCS = $(CSRCSDIR)/propagate-toz-test_Kokkos_v5.cpp
BENCHMARK = propagate_$(COMPILER)_$(MODE)
COMPILE_KOKKOS = 1
endif
ifeq ($(MODE),kokkosv6)
CSRCSDIR = kokkos_src_v6
CSRCS = $(CSRCSDIR)/propagate-toz-test_Kokkos_v6.cpp
ifneq ($(PREPIN_HOSTMEM),1)
KOKKOS_FLAGS += prepin_hostmem=$(PREPIN_HOSTMEM)
BENCHMARK = propagate_$(COMPILER)_$(MODE)
else
BENCHMARK = propagate_$(COMPILER)_$(MODE)_prepin_host
endif
COMPILE_KOKKOS = 1
endif

ifeq ($(COMPILE_KOKKOS),1)
ifneq ($(NITER),0)
KOKKOS_FLAGS += NITER=$(NITER)
else
ifeq ($(INCLUDE_DATA),0)
KOKKOS_FLAGS += NITER=100
else
KOKKOS_FLAGS += NITER=10
endif
endif
ifneq ($(NLAYER),0)
KOKKOS_FLAGS += NLAYER=$(NLAYER)
endif
ifneq ($(INCLUDE_DATA),0)
KOKKOS_FLAGS += INCLUDE_DATA=$(INCLUDE_DATA)
endif
KOKKOS_FLAGS += USE_FMAD=$(USE_FMAD)
KOKKOS_FLAGS += USE_GPU=$(USE_GPU)
ifeq ($(USE_GPU),0)
KOKKOS_FLAGS += KOKKOS_DEVICES=OpenMP
else
KOKKOS_FLAGS += KOKKOS_DEVICES=$(KOKKOS_DEVICES)
endif
ifneq ($(KOKKOS_ARCH),None)
KOKKOS_FLAGS += KOKKOS_ARCH=$(KOKKOS_ARCH)
endif
endif

################################################
# TARGET is where the output binary is stored. #
################################################
TARGET = ./bin

ADD_SUFFIX = 0
ifeq ($(MODE),cudav4)
ADD_SUFFIX = 1
endif
ifeq ($(MODE),cudav3)
ADD_SUFFIX = 1
endif
ifeq ($(MODE),cudav2)
ADD_SUFFIX = 1
endif
ifeq ($(MODE),cudav1)
ADD_SUFFIX = 1
endif
ifeq ($(MODE),cuda)
ADD_SUFFIX = 1
endif
ifeq ($(ADD_SUFFIX),1)
ifneq ($(USE_ASYNC),0)
BENCHMARK = "propagate_$(COMPILER)_$(MODE)_async"
else
BENCHMARK = "propagate_$(COMPILER)_$(MODE)_sync"
endif
else
ifeq ($(COMPILE_KOKKOS),0)
BENCHMARK = "propagate_$(COMPILER)_$(MODE)"
endif
endif

ifneq ($(MODE),omp4c)
ifneq ($(MODE),omp4cv3)
ifneq ($(MODE),omp4cv4)
ifneq ($(MODE),accc)
ifneq ($(MODE),acccv3)
ifneq ($(MODE),acccv4)
ifneq ($(MODE),accccpu)
ifneq ($(MODE),acccv3cpu)
ifneq ($(MODE),acccv4cpu)
ifeq ($(MODE),cudahyb)
CFLAGS1 += -std=c++20
else
ifeq ($(MODE),pstl)
CFLAGS1 += -std=c++20
else
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
endif
endif
endif
endif
endif


$(TARGET)/$(BENCHMARK): precmd src/$(CSRCS)
	if [ ! -d "$(TARGET)" ]; then mkdir bin; fi
	if [ $(COMPILE_CMAKE) -eq 0 ] && [ $(COMPILE_KOKKOS) -eq 0 ]; then $(CXX) $(CFLAGS1) ./src/$(CSRCS) $(CLIBS1) $(TUNE) -o $(TARGET)/$(BENCHMARK); fi
	if [ $(COMPILE_CMAKE) -eq 0 ] && [ $(COMPILE_KOKKOS) -eq 1 ]; then cd ./src/$(CSRCSDIR); make clean; make -j $(KOKKOS_FLAGS); cd ../../; fi
	if [ $(COMPILE_CMAKE) -eq 1 ]; then cd src/$(CMAKEDIR); if [ ! -d build ]; then mkdir build; fi; cd build; $(CMAKECMD); make -j32; cd ../../../; mv ./src/$(CMAKEDIR)/build/$(CMAKEOUTPUT) $(TARGET)/$(BENCHMARK); fi
	if [ -f "./cetus_output/openarc_kernel.cu" ]; then cp ./cetus_output/openarc_kernel.cu ${TARGET}/; fi
	if [ -f "./cetus_output/openarc_kernel.cl" ]; then cp ./cetus_output/openarc_kernel.cl ${TARGET}/; fi
	if [ -f "./cetus_output/openarc_kernel.hip.cpp" ]; then cp ./cetus_output/openarc_kernel.hip.cpp ${TARGET}/; fi

precmd:
	@if [ "$(CSRCS)" = "NotSupported" ]; then echo "==> [ERROR] The compiler $(COMPILER) is not supported for the mode = $(MODE)"; fi
	@if [ "$(COMPILER)" = "openarc" ]; then rm -rf ./cetus_output $(TARGET)/openarc_kernel.* $(TARGET)/*.ptx $(TARGET)/propagate_openarc_*; fi
	@if [ "$(COMPILER)" = "openarc" ]; then O2GBuild.script $(MODE) $(STREAMS) $(NITER) $(INCLUDE_DATA); fi

clean:
	rm -f $(TARGET)/$(BENCHMARK) $(TARGET)/openarc_kernel.* $(TARGET)/*.ptx *.o
	if [ $(COMPILE_KOKKOS) -eq 1 ]; then cd ./src/$(CSRCSDIR); make clean; cd ../../; fi
	if [ $(COMPILE_CMAKE) -eq 1 ]; then cd ./src/$(CMAKEDIR); rm -rf build; cd ../../; fi
	rm -rf ./src/alpaka_src_gpu/build

cleanall: purge
	rm -f $(TARGET)/* 

purge: clean
	rm -rf cetus_output openarcConf.txt 
