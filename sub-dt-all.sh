#!/bin/bash -l
#SBATCH --job-name="SF-2022"
#SBATCH --partition=RGZN8
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCiH --output=jobout/job.%j.out
#SBATCH --error=jobout/joberror.%j.out
#SBATCH --gpus=0


source activate py310
python main_DT.py
