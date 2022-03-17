icc -O3 -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=32 -Dnlayer=20 -Dnthreads=10 -o propagate-tor-test_tbb.exe ./src/propagate-tor-test_tbb.cpp -qtbb
