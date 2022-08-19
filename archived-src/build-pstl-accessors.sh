# -gpu=fastmath

nvc++ -Iinclude -O2 -std=c++17 -stdpar=gpu -gpu=cc86 -gpu=managed -gpu=fma -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll -o propagate-tor-test_pstl_cuda_nvc++_N5_opt src/propagate-tor-test_pstl_accessors.cpp -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20
