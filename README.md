# Kpdocker Introduction

KDP toolchain is a set of software which provide inputs and simulate the operation in the hardware KDP 520, 720, 530, 630 and 730. 
For better environment compatibility, the `Kpdocker` is provided which include all the dependencies as well as the toolchain software.

## 1. Toolchain Docker Deployment

### 1.1 Toolchain Docker TAGs

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

## 3. Workflow for Yolo Example

Here we take the Yolo Example to break down the workflow. 

The followings are Official Documents related for YOLOv3: 
* `YOLOv3 Step by Step`: https://doc.kneron.com/docs/#toolchain/appendix/yolo_example
* `YOLOv3 with In-Model-Preprocess trick Step by Step`: https://doc.kneron.com/docs/#toolchain/appendix/yolo_example_InModelPreproc_trick

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
m = ktc.onnx_optimizer.keras2onnx_flow('yolov3-tiny.h5', optimize=0, input_shape=[1,416,416,3])
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
```

### 3.5 Onnx Optimization

```python
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
onnx.save(m, 'yolov3-tiny.opt.onnx')
```

### 3.6 IP Evaluate

#### 3.6.1 Preprocess() Function

```python
def preprocess(pil_img):
    model_input_size = (416, 416)
    boxed_image = letterbox_image(pil_img, model_input_size)
    np_data = np.array(boxed_image, dtype='float32')
    if IN_MODEL_PREPROCESS == True:
        np_data -= 128
    else: 
        np_data /= 255.
    np_data = ktc.convert_channel_last_to_first(np_data)
    return np_data
```

#### 3.6.2 NPU Performance Simulation

```python
km = ktc.ModelConfig(32769, "0001", '520', onnx_model=m)
eval_result = km.evaluate()
```

#### 3.6.3 NPU Performance Simulation Report

```bash
===========================================
=            report on flow status        =
===========================================

                kdp520                                                                                 general
               gen_nef     FPS batch compiler compiler frontend compiler hw info compiler_cfg cpu_node Success onnx size (MB)
category case
input    input       ✓  21.352              ✓                 ✓                ✓            ✓      N/A       ✓           33

Npu performance evaluation result:
docker_version: kneron/toolchain:v0.23.0
comments:
kdp520/input bitwidth: int8
kdp520/output bitwidth: int8
kdp520/datapath bitwidth: int8
kdp520/weight bitwidth: int8
kdp520/ip_eval/fps: 21.352
kdp520/ip_eval/ITC(ms): 46.8341 ms
kdp520/ip_eval/RDMA bandwidth GB/s: 0.800000
kdp520/ip_eval/WDMA bandwidth GB/s: 0.800000
kdp520/ip_eval/GETW bandwidth GB/s: 0.800000
kdp520/cpu_node: N/A
gen fx model report: model_fx_report.html

, node, node origin, type, node backend,
0, concatenate_1, concatenate_1, NPU, concatenate_1,
1, concatenate_1_KNOPT_dummy_bn_0, concatenate_1, NPU, concatenate_1_KNOPT_dummy_bn_0,
2, concatenate_1_KNOPT_dummy_bn_1, concatenate_1, NPU, concatenate_1_KNOPT_dummy_bn_1,
3, conv2d_1, conv2d_1, NPU, npu_fusion_node_conv2d_1_leaky_re_lu_1_max_pooling2d_1,
4, conv2d_10, conv2d_10, NPU, conv2d_10,
5, conv2d_11, conv2d_11, NPU, npu_fusion_node_conv2d_8_leaky_re_lu_8_KNERON_REFORMAT_next_0,
6, conv2d_12, conv2d_12, NPU, npu_fusion_node_conv2d_12_leaky_re_lu_11,
7, conv2d_13, conv2d_13, NPU, npu_fusion_node_conv2d_12_leaky_re_lu_11_KNERON_REFORMAT_next_0,
8, conv2d_2, conv2d_2, NPU, npu_fusion_node_conv2d_2_leaky_re_lu_2_max_pooling2d_2,
9, conv2d_3, conv2d_3, NPU, npu_fusion_node_conv2d_3_leaky_re_lu_3_max_pooling2d_3,
10, conv2d_4, conv2d_4, NPU, npu_fusion_node_conv2d_4_leaky_re_lu_4_max_pooling2d_4,
11, conv2d_5, conv2d_5, NPU, npu_fusion_node_conv2d_5_leaky_re_lu_5,
12, conv2d_6, conv2d_6, NPU, npu_fusion_node_conv2d_6_leaky_re_lu_6,
13, conv2d_7, conv2d_7, NPU, npu_fusion_node_conv2d_7_leaky_re_lu_7,
14, conv2d_8, conv2d_8, NPU, npu_fusion_node_conv2d_8_leaky_re_lu_8,
15, conv2d_9, conv2d_9, NPU, npu_fusion_node_conv2d_9_leaky_re_lu_9,
16, input_1_o0_scale_shift_bn, input_1_o0_scale_shift_bn, NPU, input_1_o0_scale_shift_bn,
17, leaky_re_lu_1, leaky_re_lu_1, NPU, npu_fusion_node_conv2d_1_leaky_re_lu_1_max_pooling2d_1,
18, leaky_re_lu_10, leaky_re_lu_10, NPU, npu_fusion_node_conv2d_8_leaky_re_lu_8_KNERON_REFORMAT_next_0,
19, leaky_re_lu_11, leaky_re_lu_11, NPU, npu_fusion_node_conv2d_12_leaky_re_lu_11,
20, leaky_re_lu_2, leaky_re_lu_2, NPU, npu_fusion_node_conv2d_2_leaky_re_lu_2_max_pooling2d_2,
21, leaky_re_lu_3, leaky_re_lu_3, NPU, npu_fusion_node_conv2d_3_leaky_re_lu_3_max_pooling2d_3,
22, leaky_re_lu_4, leaky_re_lu_4, NPU, npu_fusion_node_conv2d_4_leaky_re_lu_4_max_pooling2d_4,
23, leaky_re_lu_5, leaky_re_lu_5, NPU, npu_fusion_node_conv2d_5_leaky_re_lu_5,
24, leaky_re_lu_6, leaky_re_lu_6, NPU, npu_fusion_node_conv2d_6_leaky_re_lu_6,
25, leaky_re_lu_7, leaky_re_lu_7, NPU, npu_fusion_node_conv2d_7_leaky_re_lu_7,
26, leaky_re_lu_8, leaky_re_lu_8, NPU, npu_fusion_node_conv2d_8_leaky_re_lu_8,
27, leaky_re_lu_9, leaky_re_lu_9, NPU, npu_fusion_node_conv2d_9_leaky_re_lu_9,
28, max_pooling2d_1, max_pooling2d_1, NPU, npu_fusion_node_conv2d_1_leaky_re_lu_1_max_pooling2d_1,
29, max_pooling2d_2, max_pooling2d_2, NPU, npu_fusion_node_conv2d_2_leaky_re_lu_2_max_pooling2d_2,
30, max_pooling2d_3, max_pooling2d_3, NPU, npu_fusion_node_conv2d_3_leaky_re_lu_3_max_pooling2d_3,
31, max_pooling2d_4, max_pooling2d_4, NPU, npu_fusion_node_conv2d_4_leaky_re_lu_4_max_pooling2d_4,
32, max_pooling2d_5, max_pooling2d_5, NPU, max_pooling2d_5,
33, max_pooling2d_6, max_pooling2d_6, NPU, max_pooling2d_6,
34, up_sampling2d_1, up_sampling2d_1, CPU, cpu_fusion_node_up_sampling2d_1,
```

### 3.7 Onnx Model Check

#### 3.7.1 Postprocess

```python
def postprocess(inf_results, ori_image_shape):
    tensor_data = [tf.convert_to_tensor(data, dtype=tf.float32) for data in inf_results]
    anchors_path = 'yolov3-tiny.anchors'
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    num_classes = CLASSES
    boxes, scores, classes = yolo_eval(tensor_data, anchors, num_classes, ori_image_shape)
    with tf.compat.v1.Session() as sess:
        boxes = boxes.eval()
        scores = scores.eval()
        classes = classes.eval()
    return boxes, scores, classes
```

#### 3.7.2 E2E simulator Inference

```python
input_image = Image.open('000000350003.jpg')
in_data = preprocess(input_image)
input_image.close()
out_data = ktc.kneron_inference([in_data], onnx_file='yolov3-tiny.opt.onnx', input_names=['input_1_o0'])
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)
```

#### 3.7.3 E2E simulator Inference Result

```bash
(array([[258.89148,470.26517,297.0268,524.3218],
[233.60538,218.18251,306.83316,381.80396]],dtype=float32),array([0.9251516,0.787214],dtype=float32),array([2,7],dtype=int32))
```

### 3.8 Fix Point Analysis

#### 3.8.1 Normalize All Images from a List of Test Pictures

```python
img_list = []
with open ('test_image10.txt', "r") as myfile:
    lines = myfile.read().splitlines()
for item in lines:
    image = Image.open(item)
    img_data = preprocess(image)
    img_list.append(img_data)
    image.close()
```

#### 3.8.2 Analysis

```python
bie_model_path = km.analysis({'input_1_o0': img_list})
print("\nFix point analysis done. Save bie model to '" + str(bie_model_path) + "'")
```

```bash
 Success for model "input/input" when running "general/Success"


===========================================
=            report on flow status        =
===========================================

               kdp520                                                                                                                                                             general
               knerex gen_nef     FPS batch compiler compiler frontend compiler hw info compiler_cfg cpu_node dp analysis buf (KB) dp analysis total (KB) dp_analysis result (KB) Success onnx size (MB)
category case
input    input      ✓       ✓  21.352              ✓                 ✓                ✓            ✓      N/A                21633                  21773                     140       ✓             33




Fix point analysis done. Save bie model to '/data1/kneron_flow/input.kdp520.scaled.bie'
```

### 3.9 Bie Model Check

### 3.9.1 Bie Model Generation

```python
input_image = Image.open('000000350003.jpg')
in_data = preprocess(input_image)
input_image.close()
out_data = ktc.kneron_inference([in_data], bie_file=bie_model_path, input_names=['input_1_o0'], platform=520)
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)
```
### 3.9.2 Bie Model Inference Result

```bash
(array([[260.754 ,471.40704,295.3402,522.4468],
[234.3568,210.12952,307.17825,389.8782]],dtype=float32),array([0.89978975,0.760541],dtype=float32),array([2,7],dtype=int32))
```

### 3.10 Compile

#### 3.10.1 Compile

```python
nef_model_path = ktc.compile([km])
```

#### 3.10.2 Result

```bash
[tool][info][batch_compile.cc:513][BatchCompile] compiling input.kdp520.scaled.bie
[common][info][compile.cc:62][LoadCfg] Loading config.
[piano][info][graph_gen.cc:151][GraphOptFE] Graph is generated from optimized graph, skip graph optimization
[common][info][tgt_graph_gen.cc:31][GraphOptBE] Working on hardware relevant optimizations.
[common][info][knerex_fp.cc:26][ReWriteKnerexInfo] radix_json specified, rewrite FP info.
[graph][info][knerex_info_patch.cc:180][PatchKnerexInfo] patch FP info.
[common][info][tgt_graph_gen.cc:43][GraphOptBE] Working on graph post process
[common][info][compile.cc:250][CompileImpl] Lowering IR
[cmd][info][cmd_node_gen.cc:701][CutAnalysis] Cutting image
[fmcut][info][img_cut.cc:65][ImageCut] Start image cut with mode [default]
[common][info][compile.cc:253][CompileImpl] Generating weight
[common][info][compile.cc:257][CompileImpl] Generating command
[cmd][info][cmd_generator.cc:395][GenCmds] Generate commands based on [45] cmd nodes
[cmd][info][cmd_generator.cc:1377][RemoveRedundantConfCmd] 949 out of 2108 CONF cmds are optimized out
[common][info][compile.cc:276][CompileImpl] input_size [692224], wt_size [9852368], cmd_size [5120], dram_size [1038336], sram_size [524288], fw_code_size [460]
info:
  dram_start: 1610612736
  dram_size: 1038336
  cmd_start: 0
  cmd_size: 5120
  input_start: 3145728
  input_size: 692224
  fw_code_start: 1610612736
  fw_code_size: 460
  input_num: 0
  output_num: 2
  output_size: 265200
  output_start: 8388608
[common][info][compile.cc:307][CompileImpl] Compilation completed.
[tool][info][batch_compile.cc:551][LayoutBins] Re-layout binaries
[tool][info][batch_compile.cc:601][LayoutBins] output start: 0x600e9bf0, end: 0x600e9bf0
[tool][info][batch_compile.cc:513][BatchCompile] compiling input.kdp520.scaled.bie
[common][info][compile.cc:62][LoadCfg] Loading config.
[piano][info][graph_gen.cc:151][GraphOptFE] Graph is generated from optimized graph, skip graph optimization
[common][info][tgt_graph_gen.cc:31][GraphOptBE] Working on hardware relevant optimizations.
[common][info][knerex_fp.cc:26][ReWriteKnerexInfo] radix_json specified, rewrite FP info.
[graph][info][knerex_info_patch.cc:180][PatchKnerexInfo] patch FP info.
[common][info][tgt_graph_gen.cc:43][GraphOptBE] Working on graph post process
[common][info][compile.cc:250][CompileImpl] Lowering IR
[cmd][info][cmd_node_gen.cc:701][CutAnalysis] Cutting image
[fmcut][info][img_cut.cc:65][ImageCut] Start image cut with mode [default]
[common][info][compile.cc:253][CompileImpl] Generating weight
[common][info][compile.cc:257][CompileImpl] Generating command
[cmd][info][cmd_generator.cc:395][GenCmds] Generate commands based on [45] cmd nodes
[cmd][info][cmd_generator.cc:1377][RemoveRedundantConfCmd] 949 out of 2108 CONF cmds are optimized out
[common][info][compile.cc:276][CompileImpl] input_size [692224], wt_size [9852368], cmd_size [5120], dram_size [1038336], sram_size [524288], fw_code_size [460]
info:
  dram_start: 1611570160
  dram_size: 1038336
  cmd_start: 1612608496
  cmd_size: 5120
  input_start: 1610612736
  input_size: 692224
  fw_code_start: 1622465984
  fw_code_size: 460
  input_num: 0
  output_num: 2
  output_size: 265200
  output_start: 1611304960
[common][info][compile.cc:307][CompileImpl] Compilation completed.
[tool][info][batch_compile.cc:711][CombineAllBin] Combine all bin files of all models into all_models.bin
[tool][info][batch_compile.cc:787][WriteFwInfo] Generate firmware info to fw_info.txt & fw_info.bin
[tool][info][batch_compile.cc:653][VerifyOutput]
=> 1 models
[tool][info][batch_compile.cc:661][VerifyOutput]      id: 32769
[tool][info][batch_compile.cc:662][VerifyOutput]      version: 0x1
[tool][info][batch_compile.cc:667][VerifyOutput]      addr: 0x60000000, size: 0xa9000
[tool][info][batch_compile.cc:667][VerifyOutput]      addr: 0x600a9000, size: 0x40bf0
[tool][info][batch_compile.cc:667][VerifyOutput]      addr: 0x600e9bf0, size: 0xfd800
[tool][info][batch_compile.cc:667][VerifyOutput]      addr: 0x601e73f0, size: 0x1400
[tool][info][batch_compile.cc:667][VerifyOutput]      addr: 0x601e87f0, size: 0x9655d0
[tool][info][batch_compile.cc:667][VerifyOutput]      addr: 0x60b4ddc0, size: 0x1cc
[tool][info][batch_compile.cc:670][VerifyOutput]

[tool][info][batch_compile.cc:674][VerifyOutput]   end addr 0x60b4df8c,
[tool][info][batch_compile.cc:676][VerifyOutput] total bin size 0x966b9c
[tool][info][batch_compile.cc:1233][main] batch_compile complete[0]

Compile done. Save Nef file to '/data1/kneron_flow/models_520.nef'
```

### 3.11 Knef Model Ckeck

#### 3.11.1 Knef Model Inference

```python
input_image = Image.open('000000350003.jpg')
in_data = preprocess(input_image)
input_image.close()
out_data = ktc.kneron_inference([in_data], nef_file=nef_model_path, platform=int(DEVICE), input_names=[IMPUT_NAMES])
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)
```

#### 3.11.2 Knef Model Inference Result

```bash
(array([[260.754,471.40704,295.3402,522.4468],
[234.3568,210.12952,307.17825,389.8782]],dtype=float32),array([0.89978975,0.760541],dtype=float32),array([2,7],dtype=int32))
```

### 3.12 Copy the Compiled Model from Docker to Host

```bash
(base) root@48b04a35dc1a:/workspace/examples/darknet# cp -fv /data1/kneron_flow/models_520.nef /docker_mount
'/data1/kneron_flow/models_520.nef' -> '/docker_mount/models_520.nef'
```

After exit our own `Docker`, the copied Model (`models_520.nef`) can be found in `/mnt/kpdocker` . 

## 4. Run Time Test

First, we have to plug the `KL520` dongle into the `Notebook` USB port; then follow the steps below:

### 4.1 Replace The Model in `kneron_plus` related Path. 

```
cd C:\msys64\home\kneron_plus\res\models\KL520\tiny_yolo_v3
```

### 4.2 Run the `kl520_demo_cam_generic_image_inference_drop_frame`

```
cd C:\msys64\home\kneron_plus\build\bin
```

```
C:\msys64\home\kneron_plus\build\bin>kl520_demo_cam_generic_image_inference_drop_frame.exe
```

# Quiz

#### 1. Why choose `v0.23.0` instead of `latest` ? 

#### 2. While coding python, can we use <TAB> to replace the leading space? 

#### 3. How to Convert YOLOv3 weights to ONNX? 

#### 4. While invoking a python function with a list of parameters, could we change the order of the given list? 

#### 5. Could we use ${PYTHONPATH} to replece `sys.path.append(str(pathlib.Path("/data1/keras_yolo3").resolve()))` described above?

#### 6. Why we set const `IMPUT_NAMES` to `'input_1_o0'` ? How the given `'input_1_o0'` to be determined? 









