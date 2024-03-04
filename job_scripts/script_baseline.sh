#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --export=none
#SBATCH --job-name=baseline
#SBATCH --output=job_outputs/baseline.txt

module load miniconda3/23.5.2/gcc-13.2.0

pip install -r requirements.txt
# activate the env
source activate /gpfs/users/guptar/.conda/envs/baseline

echo "Installations done"

for dataset in ship drums ficus hotdog lego materials mic
do
    for k in 5 10 15 20 25
    do
        python baseline_intrinsic.py $k 200 $dataset
    done
done
