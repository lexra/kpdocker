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
DOCKER_MOUNT=/work/docker_mount
[ ! -d ${DOCKER_MOUNT} ] && (sudo mkdir -p ${DOCKER_MOUNT} && sudo chmod 7777 ${DOCKER_MOUNT})

TAG=kpdocker
#REPOSITORY=kneron/toolchain
REPOSITORY=lexra/${TAG}
sudo docker rmi -f $(sudo docker images -a | grep "^${REPOSITORY}" | awk '{print $3}') || true

###########################################################
# docker run
###########################################################
pip3 install gdown==5.1.0
[ ! -e examples/wheelchair/datasets.zip ] && gdown https://drive.google.com/uc?id=1uSpN-bDlX9wG66K36yuscewB58pFnpbz -O examples/wheelchair/
[ ! -e examples/wheelchair/wheelchair.weights ] && gdown https://drive.google.com/uc?id=1K2fzXOUwuBjdBll3pHaldvqV41Rujsa_ -O examples/wheelchair/
rm -rfv examples/wheelchair/*tmp

docker build -t="${REPOSITORY}" .
docker run --rm -it -v ${DOCKER_MOUNT}:/docker_mount --name kpdocker \
	--shm-size=4gb \
	--gpus all --runtime=nvidia \
	${REPOSITORY}

exit 0
