#!/bin/sh

#qsub -q mamba -l nodes=1:ppn=16 -d $(pwd) bench_cpu.sh
sbatch --partition=Centaurus --nodes=1 --ntasks-per-node=16 --time=00:20:00 bench_cpu.sh
