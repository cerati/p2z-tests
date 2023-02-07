KOKKOS_PATH ?= ${KOKKOS_ROOT}
#KOKKOS_DEVICES = "OpenMP"
KOKKOS_DEVICES ?= "Cuda"
# KOKKOS_DEVICES = "OpenMP,Cuda"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

ifndef NITER
NITER=10
endif
ifndef NLAYER
NLAYER=20
endif
INCLUDE_DATA ?= 1
ifneq ($(INCLUDE_DATA),0)
TUNE += -Dinclude_data=$(INCLUDE_DATA)
endif
TUNE += -I../

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3 -DNITER=${NITER} -Dnlayers=${NLAYER} ${TUNE}
# KOKKOS_ARCH = "HSW,Kepler35"
#KOKKOS_ARCH = "SKX,Volta70"
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
EXE = "../../bin/propagate_nvcc_kokkosv3"
USE_GPU=1
else
# -9 needed on apollo
CXX = g++
CXXFLAGS = -O3 -DNITER=${NITER} -Dnlayers=${NLAYER} ${TUNE}
#CXX = g++-9
# CXX = icpc
EXE = "../../bin/propagate_gcc_kokkosv3"
KOKKOS_ARCH = "SKX"
#KOKKOS_ARCH = "POWER9"
endif

LINK = ${CXX}
LINKFLAGS =

# CXXFLAGS += -march=skylake-avx512 -qopt-zmm-usage=high

ifdef USE_GPU
CXXFLAGS += -DUSE_GPU=1
endif

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

purge: clean
	rm -f $(EXE)

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
