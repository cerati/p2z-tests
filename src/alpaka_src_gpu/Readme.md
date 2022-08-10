# alpaka p2z
## For the regular CPU version
Using Makefile on UO apollo, requires alpaka/include in the path

cd ~/p2z-tests/

make COMPILER=gcc MODE=alpaka

./bin/propagate_gcc_alpaka 

## For the GPU & CPU version 
Using Cmake options, requires alpaka installation (v0.8), cmake, and cudatoolkit

cd ~/p2z-tests/src/alpaka_src_gpu/build

source config.sh

make

./alpaka_src_gpu
