#!/bin/bash

for i in {42990..43008}; 
do 
    file="slurm-${i}.out" 
    grep -e 'Dataset' -e 'Solve time' -e 'Residual Norm' -e 'Iterations' $file
    s=$(printf "%-30s" "*")
    echo "${s// /*}"
done
