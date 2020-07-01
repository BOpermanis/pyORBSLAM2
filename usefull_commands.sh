#!/usr/bin/env bash

#docker run -it pyorbslam0 /bin/bash

docker run --name pyorbslam1 -p 2222:2222 -itd \
           -e USER=bruno -e PASSWORD=www \
           -v slam_data:/home/slam_data \
           --privileged=true \
           --device=/dev:/dev \
           pyorbslam0