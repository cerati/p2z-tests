cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=/home/cong2/fermi/p2r-tests/alpaka/install -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_CUDA_COMPILER=nvcc -DALPAKA_CXX_STANDARD=17     -DALPAKA_ACC_GPU_CUDA_ENABLE=on -DCMAKE_CUDA_ARCHITECTURES=70 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=on ..

