# KOKKOS_PATH = ${HOME}/kokkos-simd
KOKKOS_PATH = ${HOME}/kokkos
#KOKKOS_DEVICES = "OpenMP"
#KOKKOS_DEVICES = "Cuda"
 KOKKOS_DEVICES = "OpenMP,Cuda"
#EXE_NAME = "test"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
#EXE = ${EXE_NAME}.cuda
# KOKKOS_ARCH = "HSW,Kepler35"
KOKKOS_ARCH = "BDW,Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
EXE = "../../bin/propagate_nvcc_kokkos"
else
# -9 needed on apollo
CXX = g++
#CXX = g++-9
# CXX = icpc
#EXE = ${EXE_NAME}.host
EXE = "../../bin/propagate_gcc_kokkos"
KOKKOS_ARCH = "BDW"
endif

CXXFLAGS = -O3 -g --std=c++11
LINK = ${CXX}
LINKFLAGS =

# CXXFLAGS += -march=skylake-avx512 -qopt-zmm-usage=high

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
