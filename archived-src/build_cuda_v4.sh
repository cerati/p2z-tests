
#--use_fast_math for approx. versions:
#    Flush-to-zero mode is enabled (that is, denormal support is disabled)
#    Single-precision reciprocal, division, and square root are switched to approximate versions
#    Certain standard math functions are replaced by equivalent, lower-precision, intrinsics


nvcc -I./include -arch=sm_86 -O3 --extended-lambda --expt-relaxed-constexpr --use_fast_math --default-stream per-thread -std=c++17 ./src/propagate-tor-test_cuda_v4.cu -L -lcudart   -o ./"propagate_nvcc_cuda" -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=32 -Dnlayer=20 -Dnthreads=1 -Dnum_streams=4 -Dthreadsperblockx=32 -DEXCLUDE_H2D_TRANSFER -DEXCLUDE_D2H_TRANSFER

