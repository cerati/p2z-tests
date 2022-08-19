nvc++ -Iinclude -O2 -std=c++17 -stdpar=multicore -o propagate-tor-test_pstl_cuda_nvc++_N5_native src/propagate-tor-test_pstl_native.cpp -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20
