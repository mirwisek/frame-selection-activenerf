#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --export=none
#SBATCH --job-name=drums
#SBATCH --output=job_outputs/clustering.txt

module load miniconda3/23.5.2/gcc-13.2.0
# activate the env
pip install -r requirements.txt

source activate /gpfs/users/guptar/.conda/envs/myenv

for dataset in ship drums ficus hotdog lego materials mic
do
    for k in 5 10 15 20 25
    do
        python clustering_script.py $k nerf_synthetic $dataset
    done
done