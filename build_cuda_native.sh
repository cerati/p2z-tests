#--use_fast_math for approx. versions:
#    Flush-to-zero mode is enabled (that is, denormal support is disabled)
#    Single-precision reciprocal, division, and square root are switched to approximate versions
#    Certain standard math functions are replaced by equivalent, lower-precision, intrinsics

nvcc -arch=sm_86 -O3 --extended-lambda --expt-relaxed-constexpr --use_fast_math --default-stream per-thread -std=c++17 ./src/propagate-tor-test_cuda_native.cu -L -lcudart   -o ./"propagate_nvcc_cuda_native"

