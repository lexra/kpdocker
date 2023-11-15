# Kp docker 介紹

## 1. Toolchain Docker Deployment

### 1.1 Toolchain Docker TAGS

```bash
curl https://registry.hub.docker.com/v2/repositories/kneron/toolchain/tags \
    | yq -p json -o yaml | grep ' name: '
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

### 1.2 Pull the Docker Image and Login to the Docker

```bash
docker pull kneron/toolchain:v0.23.0
mkdir -p /mnt/docker
docker run --rm -it -v /mnt/docker:/docker_mount kneron/toolchain:v0.23.0
```

### 1.3 Use Our Own Docker Image

### 1.3.1 Prepare the Dockerfile

```bash
FROM kneron/toolchain:v0.23.0

RUN apt update
RUN apt install -y vim p7zip-full p7zip-rar iputils-ping net-tools udhcpc cython rar libsqlite3-dev
RUN apt install -y dirmngr --install-recommends
RUN /workspace/miniconda/bin/pip install gdown

###########################################################
# keras-yolo3
###########################################################
RUN cd /data1 && git clone https://github.com/qqwweee/keras-yolo3.git keras_yolo3 && cd -
```
### 1.3.2 Docker build and Login to Our Own Docker

```
export DOCKER_MOUNT=/mnt/kpdocker
docker build -t="kneron/toolchain:vim" .
mkdir -p ${DOCKER_MOUNT}
docker run --rm -it -v ${DOCKER_MOUNT}:/docker_mount kneron/toolchain:vim
```

## 2. Examples

### 2.1 Darknet Weights

#### 2.1.1 Dockerfile

```bash
RUN mkdir -p examples/darknet
COPY examples/darknet/compile.py examples/darknet
COPY examples/darknet/yolov3-tiny.cfg examples/darknet
COPY examples/darknet/test_image10.txt examples/darknet
RUN cd examples/darknet && cat yolov3-tiny.cfg | grep anchors | tail -1 | awk -F '=' '{print $2}' \
    > yolov3-tiny.anchors && cd -
RUN cd examples/darknet && wget https://pjreddie.com/media/files/yolov3-tiny.weights && cd -
RUN cd examples/darknet && wget http://doc.kneron.com/docs/toolchain/res/test_image10.zip \
    && unzip test_image10.zip && \
    cp /workspace/E2E_Simulator/app/test_image_folder/yolo/000000350003.jpg . && cd -
RUN cd examples/darknet && /workspace/miniconda/bin/python /data1/keras_yolo3/convert.py \
    yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.h5 && cd -
```

#### 2.1.2 Login to Our Own Docker and Compilation

```bash
...
 => exporting layers
 => writing image sha256:35e3c0feaa4c776b7f780fc62c17dbb8395d3697080ee10d9c0a35ff8e4ff269
 => naming to docker.io/kneron/toolchain:vim
(base) root@3afddac34919:/workspace#
```
```bash
cd examples/darknet
python compile.py 520
```

### 2.2 Wheelchair Weights

#### 2.2.1 Dockerfile

```bash
RUN mkdir -p examples/wheelchair
COPY examples/wheelchair/push_wheelchair.jpg examples/wheelchair
COPY examples/wheelchair/compile.py examples/wheelchair
COPY examples/wheelchair/test.txt examples/wheelchair
RUN cd examples/wheelchair && /workspace/miniconda/bin/gdown --id 1uSpN-bDlX9wG66K36yuscewB58pFnpbz \
    && unzip -o datasets.zip && rm -rfv datasets.zip && cd -
RUN cd examples/wheelchair && /workspace/miniconda/bin/gdown --id 1K2fzXOUwuBjdBll3pHaldvqV41Rujsa_ \
    && cd -
COPY examples/wheelchair/wheelchair.cfg examples/wheelchair
RUN cd examples/wheelchair && cat wheelchair.cfg | grep anchors | tail -1 | awk -F '=' '{print $2}' \
    > wheelchair.anchors && cd -
RUN cd examples/wheelchair && /workspace/miniconda/bin/python /data1/keras_yolo3/convert.py \
    ./wheelchair.cfg ./wheelchair.weights ./wheelchair.h5 && cd -
```

#### 2.2.2 Login to Our Own Docker and Compilation

```bash
...
 => exporting layers
 => writing image sha256:35e3c0feaa4c776b7f780fc62c17dbb8395d3697080ee10d9c0a35ff8e4ff269
 => naming to docker.io/kneron/toolchain:vim
(base) root@3afddac34919:/workspace#
```
```bash
cd examples/wheelchair
python compile.py 520
```

### 2.3 Freihand2d Onnx

#### 2.3.1 Dockerfile

```bash
RUN rm -rfv /data1/voc_data50
RUN mkdir -p examples/freihand2d
COPY examples/freihand2d/latest_kneron_optimized.onnx examples/freihand2d
COPY examples/freihand2d/compile.py examples/freihand2d
RUN wget https://www.kneron.com/forum/uploads/112/SMZ3HLBK3DXJ.7z -O SMZ3HLBK3DXJ.7z && \
    7z x SMZ3HLBK3DXJ.7z -o/workspace/examples/freihand2d && rm -rfv SMZ3HLBK3DXJ.7z
```

#### 2.3.2 Login to Our Own Docker and Compilation

```bash
...
 => exporting layers
 => writing image sha256:35e3c0feaa4c776b7f780fc62c17dbb8395d3697080ee10d9c0a35ff8e4ff269
 => naming to docker.io/kneron/toolchain:vim
(base) root@3afddac34919:/workspace#
```
```bash
cd examples/freihand2d
python compile.py
```

## 3. Workflow

### 3.1 Import

```
import ktc
import os
import onnx
from PIL import Image
import numpy as np
import re

import tensorflow as tf
import pathlib
import sys
sys.path.append(str(pathlib.Path("/data1/keras_yolo3").resolve()))
from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image
```

### 3.2 Const Values to be checked

```python
CWD = '/workspace/examples/darknet/'
NAME = 'yolov3-tiny'
IN_MODEL_PREPROCESS = True
TEST_LIST = 'test_image10.txt'
TEST_PICTURE = '000000350003.jpg'
IMPUT_NAMES = 'input_1_o0'

DEVICE = '520'
CLASSES = 80
CHANNELS = 3
WIDTH = 416
HEIGHT = 416
```

CWD: Acronym for `Current Working Directory`

NAME: Darknet Cfg File Prefix

TEST_LIST: Test Pictures List

TEST_PICTURE: One Picture for Test

### 3.3 Model Conversion

Convert yolov3 weights to Keras h5. 

```bash
/workspace/miniconda/bin/python /data1/keras_yolo3/convert.py \
    yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.h5
```

### 3.4 Convert from Keras to Onnx

```python
m = ktc.onnx_optimizer.keras2onnx_flow('yolov3-tiny.h5', optimize=0, input_shape=[1,WIDTH,HEIGHT,CHANNELS])
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
```

### 3.5 Onnx Optimization

```python
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
onnx.save(m, 'yolov3-tiny.opt.onnx')
```

### 3.6 IP Evaluate (NPU Performance Simulation)

```python
km = ktc.ModelConfig(32769, "0001", DEVICE, onnx_model=m)
eval_result = km.evaluate()
```












