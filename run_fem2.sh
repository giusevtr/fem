#!/bin/bash

# python param_search_adult.py small 0.1 0.5 1 newadafem --samples_c=200 --eps0_min=0.00001 --eps0_max=0.005 --noise_min=0.5 --noise_max=5
#declare -a epsarr="0.003 0.005 0.007 0.009";
declare -a epsarr="0.006 0.007 0.008 0.009 0.01";
declare -a noise="0.1 0.12 0.14 0.16 0.18";
noise=
eps0=
#    python generate.py loans 24 5 0.1 0.15 0.20 0.25 0.5 1  newadafem --samples=100 --noise="$n" --eps0="$eps0"
python fem.py adult 64 3 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.16 --epsilon_split=0.008 # 0.008 0.16 50
python fem.py adult 64 5 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.16 --epsilon_split=0.010 # 0.01 0.16 50
python fem.py loans 64 3 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.10 --epsilon_split=0.006 # 0.006 0.1 50
python fem.py loans 64 5 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.14 --epsilon_split=0.007 # 0.007 0.14 50