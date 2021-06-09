#!/bin/bash

# python param_search_adult.py small 0.1 0.5 1 newadafem --samples_c=200 --eps0_min=0.00001 --eps0_max=0.005 --noise_min=0.5 --noise_max=5
#declare -a epsarr="0.003 0.005 0.007 0.009";
declare -a epsarr="0.006 0.007 0.008 0.009 0.01";
declare -a noise="0.1 0.12 0.14 0.16 0.18";
declare -a workload="32 128 256 512 1024";

for n in ${noise[@]};
do
  for eps0 in ${epsarr[@]};
  do
    for w in ${workload[@]};
    do
       python fem.py adult "$w" 3 0.1 --samples=50 --noise_multiple="$n" --epsilon_split="$eps0"
       python fem.py adult "$w" 5 0.1 --samples=50 --noise_multiple="$n" --epsilon_split="$eps0"
       python fem.py loans "$w" 3 0.1 --samples=50 --noise_multiple="$n" --epsilon_split="$eps0"
       python fem.py loans "$w" 5 0.1 --samples=50 --noise_multiple="$n" --epsilon_split="$eps0"

    done
  done
done
