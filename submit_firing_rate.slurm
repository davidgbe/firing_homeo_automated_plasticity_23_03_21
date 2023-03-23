#!/bin/bash

## Presets
#SBATCH --job-name=firing_rate

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=10
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python train_firing_rate.py --std_expl 3 --pool_size 10 --batch 10 --fixed_data 1 --l1_pen L1_PENALTY --input_current_type INPUT_CURRENT_TYPE --seed SEED

deactivate

## Exit
exit 0
