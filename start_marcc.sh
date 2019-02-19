#!/bin/bash -l
#SBATCH --job-name=hasher_geom
#SBATCH --time=0-12:0:0
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --mail-type=end
#SBATCH --mail-user=bjiang8@jhu.edu
#SBATCH --export=ALL

echo "Running on $SLURMD_NODENAME ..."
echo "Slurm job ID = $SLURM_JOB_ID"

module load pytorch cuda/9.0

export PATH=$HOME/.local/bin/:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages/:$PYTHONPATH

python $HOME/DeepHasher/hashingNet_geom.py

