CC = icpc -g

MKL_LINK =  -L${MKLROOT}/lib/intel64 -mkl=parallel -liomp5 -lpthread -lm -ldl
# MKL_LINK =  -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl #
# MKL_LINK = -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm -ldl #
# MKL_OPT  = -DMKL_ILP64 -I${MKLROOT}/include #
MKL_OPT  = -I${MKLROOT}/include #

# CBLAS_LINK = -L/home/users/gravelle/soft/OpenBLAS/lib -lopenblas -lpthread
# CBLAS_OPT  = -static -I/home/users/gravelle/soft/OpenBLAS/include

OPT  = -O3
OPT += -std=c++17 
# OPT += -qopt-report
OPT += -march=skylake-avx512
OPT +=  -qopt-zmm-usage=high
OPT +=  -fopenmp
# OPT += -no-vec 
# OPT += -DUSE_CALI
# OPT += -I${CALIPER_DIR}/include
# OPT += -L${CALIPER_DIR}/lib
# OPT +=  -lcaliper 

all: mkl_p2z

mkl_p2z:
		$(CC) ptoz_data.cpp	ptoz_mkl.cpp -o test_mkl.out -fopenmp $(OPT)  $(MKL_OPT) $(MKL_LINK) 

.PHONY: clean
clean:
		rm -f *.log *.o *.dot *.optrpt *.out


