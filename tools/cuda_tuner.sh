#!/bin/bash

source ../../mkFit/xeon_scripts/init-env.sh

#rm tuner_output.txt

trktunes=(9600) 
#trktunes=(9600 12800 19200 38400 64000 96000)
#tunes=(1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150)

#btunes=(1 2 4 8 16 32 64 128)
btunes=(128)
compilers=("nvcc")
modes=("cuda")
#threadsx=(1 2 4 8 16 32 64 128 256 512 1024)
threadsx=(1024)
#threadsy=(1 2 4 8 16 32 64 128 256 512 1028 2048 4096)
#threadsx=(1 2 4 8 16 32 128)
#threadsy=(1 2 4 8 16 32 128)
blocks=(1 2 5 7 10 12 15 17 20 25 30 40 50 60 70 80 90 100)
streams=(1 2 5 7 10 12 15 17 20 25 30 40 50)
#blocks=(20)
#streams=(10)

for idx in "${!compilers[@]}";do
  compiler=${compilers[$idx]}
  mode=${modes[$idx]}
  filename="../bin/propagate_${compiler}_${mode}"
  outname="output_tuner/cudaoutput_${compiler}_${mode}_fixedthreadstot2.txt"
  echo ${compiler} and ${mode}
  rm ${outname}
  
  for stream in ${streams[@]}; do
    for block in ${blocks[@]}; do
    for threadx in ${threadsx[@]}; do
 #   for thready in ${threadsy[@]}; do
    
      make -C .. COMPILER=${compiler} MODE=${mode} BLOCKS=${block} STREAMS=${stream} THREADSX=${threadx} clean
      make -C .. COMPILER=${compiler} MODE=${mode} BLOCKS=${block} STREAMS=${stream} THREADSX=${threadx}
      for ((i=0; i<5; i++)); do
        echo "${mode}(${compiler})" ${compiler} ${mode} ${stream} ${block} ${threadx} ${i}  
        echo -n "${mode}(${compiler})" ${compiler} ${mode} ${stream} ${block} ${threadx} ${i}' ' >> ${outname}
        ./${filename} | grep "formatted" | cut -d' ' -f2- >> ${outname}
  #    done
    done
  done
  done
  done
done




