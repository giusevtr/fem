#!/bin/bash

# python param_search_adult.py small 0.1 0.5 1 newadafem --samples_c=200 --eps0_min=0.00001 --eps0_max=0.005 --noise_min=0.5 --noise_max=5
#declare -a epsarr="0.003 0.005 0.007 0.009";
declare -a epsarr="0.006 0.007 0.008 0.009 0.01";
declare -a noise="0.1 0.12 0.14 0.16 0.18";

for n in ${noise[@]};
do
  for eps0 in ${epsarr[@]};
  do
#    python generate.py loans 24 5 0.1 0.15 0.20 0.25 0.5 1  newadafem --samples=100 --noise="$n" --eps0="$eps0"
     python fem.py adult-small 35 3  0.1 0.15 0.20 0.25 0.5 1 --samples=50 --noise_multiple="$n" --epsilon_split="$eps0"
    # python generate.py adult 65 5 0.1 0.15 0.20 1 newadafem --samples=100 --noise="$n" --eps0="$eps0"
  done
done
