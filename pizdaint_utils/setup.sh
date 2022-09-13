mkdir -p /scratch/$USER/registry
podman run \
    --name registry \
    --privileged \
    --detach \
    --restart always \
    --publish 5000:5000 \
    --volume /scratch/$USER/registry:/var/lib/registry \
    registry:2
scp $( dirname -- "$0"; )/container_job.sbatch Pizdaint:container_job.sbatch
