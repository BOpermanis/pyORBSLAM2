#!/usr/bin/env bash

#docker run -it 08e9427b5bf3 /bin/bash

docker run --name pyorb0 -p 2222:2222 -itd \
           -e USER=bruno -e PASSWORD=www \
           -v slam_data:/home/slam_data \
           --privileged=true \
           --device=/dev:/dev \
           pyorb

docker rm $(docker ps -a -f status=exited -q)
docker rmi $(docker images -a -q)
