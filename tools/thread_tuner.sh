#!/bin/bash

source ../../mkFit/xeon_scripts/init-env.sh

#rm tuner_output.txt

#trktunes=(9600) 
#trktunes=(9600 12800 19200 38400 64000 96000)
#tunes=(1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150)

nthreads=(1 2 4 8 16 32 64 128 256 512 1024)
btunes=(1 2 4 8 16 32 64 128 256 512 1024)
#btunes=(128)
compilers=("gcc" "icc" "pgi" "gcc" "icc" "gcc" "icc")
modes=("omp" "omp" "omp" "tbb" "tbb" "eigen" "eigen")
#compilers=("pgi")
#modes=("omp")
for idx in "${!compilers[@]}";do
  compiler=${compilers[$idx]}
  mode=${modes[$idx]}
  filename="../bin/propagate_${compiler}_${mode}"
  outname="output_tuner/nthreads_${compiler}_${mode}.txt"
  echo ${compiler} and ${mode}
  rm ${outname}
  
#  for trktune in ${trktunes[@]}; do
    for btune in ${btunes[@]}; do
      for nthread in ${nthreads[@]}; do
    
      make -C .. COMPILER=${compiler} MODE=${mode} NTHREADS=${nthread} TUNEB=${btune} clean
      make -C .. COMPILER=${compiler} MODE=${mode} NTHREADS=${nthread} TUNEB=${btune} 
      for ((i=0; i<5; i++)); do
        echo "${mode}(${compiler})"    ${compiler} ${mode} ${nthread} ${btune} ${i}  
        echo -n "${mode}(${compiler})" ${compiler} ${mode} ${nthread} ${btune} ${i}' ' >> ${outname}
        ./${filename} | grep "formatted" | cut -d' ' -f2- >> ${outname}
        done
      done
    done
done




