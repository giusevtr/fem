#!/bin/bash

for i in 1 2 3 4 5
do
  python fem.py adult 64 3 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.16 --epsilon_split=0.008 # 0.008 0.16 50
  python fem.py adult 64 5 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.16 --epsilon_split=0.010 # 0.01 0.16 50
  python fem.py loans 64 3 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.10 --epsilon_split=0.006 # 0.006 0.1 50
  python fem.py loans 64 5 0.1 0.15 0.2 0.25 0.5 1 --samples=50 --noise_multiple=0.14 --epsilon_split=0.007 # 0.007 0.14 50
done
