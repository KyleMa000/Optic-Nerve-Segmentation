#!/bin/bash
# The interpreter used to execute the script

# directives that convey submission options:

#SBATCH --job-name=opticnerve_classification
#SBATCH --mail-user=#####
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=28:00:00
#SBATCH --account=#####
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/3.10.4
module load pytorch
module load matplotlib

pip install -U scikit-learn
pip install -U pynrrd

# The application(s) to execute along with its input arguments and options:
python train.py