# Kp docker 介紹

## 1. Toolchain Docker Deployment

### 1.1 TAGS List

```bash
curl https://registry.hub.docker.com/v2/repositories/kneron/toolchain/tags | yq -p json -o yaml | grep ' name: '
```

```bash
    name: latest
    name: v0.23.1
    name: "720"
    name: "520"
    name: base-20230922
    name: base
    name: v0.23.0
    name: se_20230926
    name: v0.22.0
    name: v0.21.0
    name: base-20230321
    name: v0.20.2
    name: v0.20.1
    name: v0.20.0
    name: v0.19.0
    name: v0.18.2
    name: v0.18.1
    name: base-20220804
    name: v0.18.0_s0
    name: v0.18.0
    name: base-20220622
    name: base-20220615
    name: base-20220609
    name: base-20220602
    name: base-20220526
```

Here we choose `v0.23.0` instead of `latest` .

### 1.2 Pull the docker image and login to the docker

```bash
docker pull kneron/toolchain:v0.23.0
docker run --rm -it -v /mnt/kpdocker:/docker_mount kneron/toolchain:v0.23.0
```

### 1.3 Build our own docker image

### 1.3.1 Prepare the Dockerfile

```bash
FROM kneron/toolchain:v0.23.0
RUN apt update
RUN apt install -y vim p7zip-full p7zip-rar iputils-ping net-tools udhcpc cython rar libsqlite3-dev
RUN apt install -y dirmngr --install-recommends
RUN /workspace/miniconda/bin/pip install gdown
```
### 1.3.2 Docker build login to the docker

```
export DOCKER_MOUNT=/mnt/kpdocker
docker build -t="kneron/toolchain:vim" .
docker run --rm -it -v ${DOCKER_MOUNT}:/docker_mount kneron/toolchain:vim
```



