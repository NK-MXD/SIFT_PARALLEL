#!/bin/bash
#SBATCH -p wzhctest
#SBATCH --qos=low
#SBATCH -J cpu_multithread
#SBATCH -o cpu_multithreadJob
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1

source ~/.bashrc
module load compiler/cmake/3.23.1

./build/sift