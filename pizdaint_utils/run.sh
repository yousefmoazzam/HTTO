podman build . -t localhost/htto
podman push --tls-verify=false localhost/htto localhost:5000/htto 
ssh -R5000:localhost:5000 Pizdaint \
    'module load singularity; \
    singularity pull -F --nohttps $SCRATCH/htto_latest.sif docker://localhost:5000/htto:latest; \
    module remove singularity; \
    sbatch container_job.sbatch $@'