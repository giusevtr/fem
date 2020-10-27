#!/bin/bash
rm mwem_results/mwem.csv
rm mwem_results/mwem_all.csv

NUM_RUNS=5
for MARGINAL in 5 # 5
do
  for WORKLOAD in 64 # 8 16 32 64 128
  do
    for WORKLOAD_SEED in  0 # 1 2 3 4
    do
      for EPSILON in 0.1 # 0.15 0.2 0.25 0.5 0.75 1.0 #0.6 0.7 0.8 0.9 1.0 #
      do
        for SF in 5e-4 # 1e0 1e-1 5e-2 3e-2 1e-2 5e-3 1e-3 5e-4
        do
          for SF_SEED in 0 1 2 3 4
          do
            python mwem_ss.py --num_runs $NUM_RUNS --marginal $MARGINAL \
            --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
            --epsilon $EPSILON --eps0 0.003 \
            --support_frac $SF --sf_seed $SF_SEED
          done
        done
      done
    done
  done
done

# 0.003 0.005 0.007 0.009