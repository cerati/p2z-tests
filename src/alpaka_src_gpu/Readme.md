# alpaka p2z

## alpaka Installation (v0.8)
cmake and boost are required\
git clone --branch 0.8.0 https://github.com/alpaka-group/alpaka.git\
cd alpaka\
mkdir build && cd build\
cmake -DCMAKE_INSTALL_PREFIX=/install/ ..\
cmake --install .

## For the regular CPU version
Using Makefile on UO apollo requires alpaka/include to the path. Switch cpu acc in the main() function

cd ~/p2z-tests/\
make COMPILER=gcc MODE=alpaka\
./bin/propagate_gcc_alpaka

## For the GPU & CPU version 
Using Cmake options, requires cudatoolkit. Switch acc in the config.sh

cd ~/p2z-tests/src/alpaka_src_gpu/build\
source config.sh\
make\
./alpaka_src_gpu\
