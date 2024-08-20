# alpaka p2z

## alpaka Installation (v0.8)
cmake and boost are required\
git clone --branch 0.8.0 https://github.com/alpaka-group/alpaka.git\
cd alpaka\
mkdir build && cd build\
cmake -DCMAKE_INSTALL_PREFIX=/install/ ..\
cmake --install .

## For the GPU & CPU version 
Using Makefile in ${p2z-tests-root-dir} directory

cd ${p2z-tests-root-dir}
//For GPU
make COMPILER=nvcc MODE=alpaka
./bin/propagate_nvcc_alpaka

//For CPU
make COMPILER=gcc MODE=alpaka
./bin/propagate_gcc_alpaka
