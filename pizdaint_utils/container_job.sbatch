#!/bin/bash -l
#SBATCH --job-name=vusa
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=hck

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
export XDG_RUNTIME_DIR=$SCRATCH/runtime_dir

module load daint-gpu
module load singularity

singularity run $SCRATCH/htto_latest.sif \
    nsys profile \
    mpirun -n 12 \
    python -m htto $@
