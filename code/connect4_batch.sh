#!/bin/bash
#SBATCH --time=20:0:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=connect4
#SBATCH --mem=1800
module load Python/3.6.4-foss-2018a
python training_peregrine.py