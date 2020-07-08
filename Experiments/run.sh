#!/bin/bash

#python run_fem.py --dataset=loans --workload=10 --marginal=5 --epsilon=0.1
#python run_fem.py --dataset=loans --workload=20 --marginal=5 --epsilon=0.1
#python run_fem.py --dataset=loans --workload=40 --marginal=5 --epsilon=0.1
#python run_fem.py --dataset=loans --workload=80 --marginal=5 --epsilon=0.1
python run_fem.py loans 10 5 0.1
python run_fem.py loans 20 5 0.1
python run_fem.py loans 40 5 0.1
python run_fem.py loans 80 5 0.1
