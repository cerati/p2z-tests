# README for the p2z versions

this directory (src_complete) contains the different versions of the p2z benchmark.
There are several versions for CPU and GPU use:
* alpaka ---- 
* CUDA ------ 
* CUDA v2 --- 
* Eigen ----- 
* Kokkos ---- 
* OMP ------- 
* OpenACC --- 
* OpenMP4 --- 
* TBB ------- 

### Data Structures
The data is defined as a large (\~10k) set of tracks and hits each of which has 
several matrices and vectors. Each of these are no larger than 6x6 and some are 
3x3 while the vectors are length 3 or 6. Each set of matrices and vectors are 
organzized into blocks so that corresponding elements in several of the matrices
are adjecent in memory. This alignment allows for more effecient use of SIMD 
hardware when computing with small matrices.


#### Tracks


#### Hits


### Algorithms


### Versions and Compile lines

#### alpaka

* `g++  src_complete/propagate-toz-test_alpaka.cpp -I/mnt/data1/mgr85/p2z-tests/alpaka_lib/include -o ./bin/propagate_gcc_alpaka_0`
* `icc -I/include -L/lib -Wl,-rpath,/lib -ltbb -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED src_complete/propagate-toz-test_alpaka.cpp -I/mnt/data1/mgr85/p2z-tests/alpaka_lib/include  -Dnum_streams=1 -o ./bin/"propagate_icc_alpaka_0"`


#### CUDA

* `nvcc -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread src_complete/propagate-toz-test_CUDA.cu -L -lcudart   -o ./bin/"propagate_nvcc_cuda_0"`


#### CUDA v2

* `nvcc -arch=sm_70 -O3 -DUSE_GPU --default-stream per-thread -maxrregcount 64 src_complete/propagate-toz-test_CUDA_v2.cu -L -lcudart   -o ./bin/"propagate_nvcc_cudav2_0"`


#### Eigen

* `g++ -fopenmp -O3 -fopenmp-simd -lm -lgomp -march=native  src_complete/propagate-toz-test_Eigen.cpp -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub -L -lcudart   -o ./bin/"propagate_gcc_eigen_0"`
* `nvcc -arch=sm_70 --default-stream per-thread -O3 --expt-relaxed-constexpr -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub src_complete/propagate-toz-test_Eigen.cu -I/mnt/data1/dsr/mkfit-hackathon/eigen -I/mnt/data1/dsr/cub -L -lcudart   -o ./bin/"propagate_nvcc_eigen_0"`


#### OMP

* `g++ -O3 -I. -fopenmp  src_complete/propagate-toz-test_OMP.cpp -lm -lgomp  -o ./bin/"propagate_gcc_omp_0"`
* `icc -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high src_complete/propagate-toz-test_OMP.cpp   -o ./bin/"propagate_icc_omp_0"`


#### OpenACC

* `pgc++ -I. -Minfo=acc -fast -Mfprelaxed -acc -ta=tesla -mcmodel=medium -Mlarge_arrays src_complete/propagate-toz-test_OpenACC.cpp   -o ./bin/"propagate_pgi_acc_0`


#### OpenMP4

* ` gcc -O3 -I. -fopenmp -foffload="-lm" src_complete/propagate-toz-test_OpenMP4.cpp -lm -lgomp   -o ./bin/"propagate_gcc_omp4_0"`


#### TBB

* `g++ -fopenmp -O3 -I. src_complete/propagate-toz-test_tbb.cpp -I/include -L/lib -Wl,-rpath,/lib -ltbb -lm -lgomp -L/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8  -o ./bin/"propagate_gcc_tbb_0"`
* `icc -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high src_complete/propagate-toz-test_tbb.cpp -I/include -L/lib -Wl,-rpath,/lib -ltbb  -o ./bin/"propagate_icc_tbb_0"`

#### Kokkos

Kokkos has it's own makefile and Readme in the kokkos_src subdirectory refer to those for instructions

