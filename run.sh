#!/bin/bash -e

sudo echo ""

###########################################################
# TAGS
###########################################################
[ 0 -eq $(snap list | grep "^yq" | wc -l) ] && sudo snap install yq
curl https://registry.hub.docker.com/v2/repositories/kneron/toolchain/tags | yq -p json -o yaml | grep ' name: '

###########################################################
# /var/run/docker.sock file permission
###########################################################
[ -e /var/run/docker.sock ] && \
	sudo chmod 777 /var/run/docker.sock


###########################################################
# docker_mount
###########################################################
DOCKER_MOUNT=/mnt/docker
[ ! -d ${DOCKER_MOUNT} ] && (sudo mkdir -p ${DOCKER_MOUNT} && sudo chmod 7777 ${DOCKER_MOUNT})

#docker rmi -f $(docker images -aq) || true

###########################################################
# docker run
###########################################################
docker build -t="kneron/toolchain:vim" .
docker run --rm -it -v ${DOCKER_MOUNT}:/docker_mount --name kpdocker kneron/toolchain:vim

exit 0
