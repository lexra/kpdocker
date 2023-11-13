#!/bin/bash -e

#pip3 install numpy==1.23.5 || true

[ 0 -eq $(snap list | grep "^yq" | wc -l) ] && sudo snap install yq
./dockertags kneron/toolchain

#docker rmi -f $(docker images -aq) || true

docker build -t="kneron/toolchain:vim" .
docker run --rm -it -v /mnt/docker:/docker_mount kneron/toolchain:vim

#pip3 install numpy==1.19.5
exit 0
