#!/bin/sh

# qsub -q mamba -l nodes=1:ppn=7:gpus=1 -d $(pwd) test_gpu.sh
sbatch --partition=GPU --nodes=1 --ntasks-per-node=7 --gres=gpu:1 --time=00:20:00 test_gpu.sh
