#! /bin/bash
docker run \
    --mount type=bind,source=/home/mheep/Pictures,target=/mnt \
    --runtime=nvidia \
    --gpus all \
    adaptive-screen-meshing "$@"