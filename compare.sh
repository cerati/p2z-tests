#!/bin/bash

source ../mkFit/xeon_scripts/init-env.sh

rm compare_output.txt

for filename in bin/allTypes/*; do
  compiler=$(echo ${filename} | cut -f2 -d_)
  mode=$(echo ${filename} | cut -f3 -d_)
  for ((i=0; i<2; i++)); do
    echo ${compiler} ${mode} ${i}  
    echo -n ${compiler} ${mode} ${i}' ' >> compare_output.txt
    ./${filename} | grep "formatted" | cut -d' ' -f2- >> compare_output.txt
  done
done




