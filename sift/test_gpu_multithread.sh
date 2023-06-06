#!/bin/bash
#SBATCH -p wzhdtest
#SBATCH --qos=low
#SBATCH -J gpu_multithread
#SBATCH -o gpu_multithreadJob
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --gres=dcu:4
#SBATCH --exclusive

export HIP_VISIBLE_DEVICES=0,1,2,3
source ~/.bashrc
module load compiler/cmake/3.23.1

./build/sift