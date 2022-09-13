#!/bin/bash -l
#SBATCH --job-name=vusa
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --account=hck

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
export XDG_RUNTIME_DIR=$SCRATCH/runtime_dir

module load daint-gpu
module load singularity

singularity run $SCRATCH/htto_latest.sif
