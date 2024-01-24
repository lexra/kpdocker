# Kpdocker Introduction

KDP toolchain is a set of software which provide inputs and simulate the operation in the hardware KDP 520, 720, 530, 630 and 730. 
For better environment compatibility, the `Kpdocker` is provided for which we include all the dependencies as well as the toolchain software.

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
export DOCKER_MOUNT=/mnt/docker
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
docker run --rm -it -v /mnt/docker:/docker_mount kneron/toolchain:vim
```

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
RUN cd examples/wheelchair && /workspace/miniconda/bin/gdown \
    --id 1uSpN-bDlX9wG66K36yuscewB58pFnpbz \
    && unzip -o datasets.zip && rm -rfv datasets.zip && cd -
RUN cd examples/wheelchair && /workspace/miniconda/bin/gdown \
    --id 1K2fzXOUwuBjdBll3pHaldvqV41Rujsa_ && cd -
COPY examples/wheelchair/wheelchair.cfg examples/wheelchair
RUN cd examples/wheelchair && cat wheelchair.cfg \
    | grep anchors | tail -1 | awk -F '=' '{print $2}' \
    > wheelchair.anchors && cd -
RUN cd examples/wheelchair && /workspace/miniconda/bin/python /data1/keras_yolo3/convert.py \
    ./wheelchair.cfg ./wheelchair.weights ./wheelchair.h5 && cd -
```

#### 2.2.2 Login to Our Own Docker and Compilation

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
cd examples/freihand2d
python compile.py
```

## 3. Workflow for YOLOv3 Example

In the following parts of this page, you can go through the basic toolchain working process to get familiar with the toolchain.
Below is a breif diagram showing the workflow of how to generate the binary from a floating-point model using the toolchain.

<img src=https://doc.kneron.com/docs/toolchain/imgs/manual/Manual_Flow_Chart.png width=480 />

To keep the diagram as clear as possible, some details are omitted. But it is enough to show the general structure. There are three main sections:
* Floating-point model preparation. Convert the model from different platforms to onnx and optimize the onnx file. Evaluate the onnx the model to check the operator support and the estimate performance. Then, test the onnx model and compare the result with the source.
* Fixed-point model generation. Quantize the floating-point model and generate bie file. Test the bie file and compare the result with the previous step.
* Compilation. Batch compile multiple bie models into a nef format binary file. Test the nef file and compare the result with the previous step.

Here we take the Yolo Example to break down the workflow. 

The followings are official documents related to YOLOv3: 
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

#### 3.2.1 CLASSES, CHANNELS, WIDTH, HEIGHT

It's very important to check the given cfg, `yolov3-tiny.cfg`, precisely conform to the given weights, `yolov3-tiny.weights`. 
We provide function() to runtime update the necessary values into the workflow on 'classes', 'channels', 'width', 'height', from the given cfg; 
hence it's not necessary to modify the workflow source code. 

```python
def darknetKeyValue(cfg, key):
    with open(cfg, "r") as f:
        while True:
            lineN = f.readline()
            if not lineN:
                break
            pattern = r"^" + key + "\s{0,}=\s{0,}(\w+)$"
            line = lineN.rstrip()
            m = re.match(pattern, line)
            if m:
                return m.group(1)
    return None

CLASSES = int(darknetKeyValue(cfg = CWD + NAME + '.cfg', key = 'classes'))
CHANNELS = int(darknetKeyValue(cfg = CWD + NAME + '.cfg', key = 'channels'))
WIDTH = int(darknetKeyValue(cfg = CWD + NAME + '.cfg', key = 'width'))
HEIGHT = int(darknetKeyValue(key = 'height', cfg = CWD + NAME + '.cfg'))
```

### 3.3 Floating-Point Model Preparation

Convert yolov3 weights to Keras h5. 

```bash
/workspace/miniconda/bin/python /data1/keras_yolo3/convert.py \
    yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.h5
```

### 3.4 Convert from Keras to Onnx

```python
m = ktc.onnx_optimizer.keras2onnx_flow(keras_model_path='yolov3-tiny.h5', \
    optimize=0, input_shape=[1,416,416,3])
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
```

#### 3.4.1 Arguments to `ktc.onnx_optimizer.onnx2onnx_flow()`

* `keras_model_path (str)`: the input hdf5/h5 model path.
* `optimize (int, optional)`: optimization level. Defaults to 0.
* `input_shape (List, optional)`: change the input shape if set. Only single input model is supported. Defaults to None.

Returns: the converted onnx (onnx.ModelProto) .


### 3.5 Model Optimization (Onnx Optimization)

```python
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
onnx.save(m, 'yolov3-tiny.opt.onnx')
```

### 3.6 Model Evaluation (IP Evaluation)

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

#### 3.6.2 Floating-Point Model Inference (NPU Performance Simulation)

```python
km = ktc.ModelConfig(32769, "0001", '520', onnx_model=m, onnx_path=None, debug=False)
eval_result = km.evaluate()
```

##### 3.6.2.1 Arguments to `ktc.ModelConfig()`

* `id (int)`: model ID
* `version (str)`: version number which should be a four digit hex, e.g. "0a2f"
* `platform (str)`: hardware platform, should be "520" or "720"
* `onnx_model (ModelProto, optional)`: loaded onnx model. Defaults to None.
* `onnx_path (str, optional)`: onnx file path. Defaults to None.
* `bie_path (str, optional)`: bie file path. Defaults to None. One of these three parameters is required: onnx_model, onnx_path, bie_path
* `radix_json_path (str, optional)`: radix json path. Defaults to None.
* `compiler_config_path (str, optional)`: compiler config json path. Defaults to None.
* `debug (bool, optional)`: debug mode. Defaults to False.

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

#### 3.6.4 Hardware Supported Operators

Refer to the official document, <a href='https://doc.kneron.com/docs/#toolchain/appendix/operators'>Hardware Supported Operators</a> . 

It's important to check in advance if a given model (downloaded from internet) 's hidden layers' meet the required <a href='https://doc.kneron.com/docs/#toolchain/appendix/operators'>Hardware Supported Operators</a> . 

### 3.7 Floating-Point Model Inference (Onnx Model Check)

#### 3.7.1 Postprocess

```python
def postprocess(inf_results, ori_image_shape): -> boxes, scores, classes
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
out_data = ktc.kneron_inference([in_data], \
    onnx_file='yolov3-tiny.opt.onnx', input_names=['input_1_o0'])
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)
```

##### 3.7.2.1 Arguments to `ktc.kneron_inference()`

There are only two items that you need to prepare to run the inference function. Everything else is optional.

* preprocessed input: list of NumPy arrays in channel first (1, c, h, w) format. Before 0.21.0, this took in channel last format.
* model file: depending on what kind of model you want to run, but it will be one of NEF, ONNX, and BIE file

Inputs: 

* `pre_results`: same as preprocessed input mentioned above
* `nef_file/onnx_file/bie_file`: path to your input model file
  * only one of these will be used, if they are all specified, priority is NEF -> ONNX -> BIE
* `model_id`: ID of model to run inference
  * only used with NEF file if file has multiple models
* `input_names`: list of input node names
  * only needed with ONNX/BIE file
* `data_type`: string data format that you would like the output returned as
  * float or fixed
* `reordering`: list of node names/integers specifying the output order
  * integers for NEF file without ioinfo_file, node names with ioinfo_file
  * node names for ONNX and BIE file
* `ioinfo_file`: string path to file mapping output node number to name
  * only used with NEF file
* `dump`: flag to dump intermediate nodes
* `platform`: integer platform to be used
  * used with NEF file to prepare CSIM input
  * used with BIE file to indicate Dynasty fixed model version
  * 520, 530, 630, 720, 730
* `platform_version`: indicates version for a specific platform

Output: 

Output will be a list of NumPy arrays in ONNX shape format. It will be in the order specified by reordering; if reordering is not speicifed, it will be in default order provided by the model.

#### 3.7.3 E2E simulator Inference Result

```bash
(array([[258.89148,470.26517,297.0268,524.3218],
[233.60538,218.18251,306.83316,381.80396]],dtype=float32),array([0.9251516,0.787214],dtype=float32),array([2,7],dtype=int32))
```

### 3.8 Fixed-Point Model Generation (Fix Point Analysis)

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

#### 3.8.2 Quantization

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

#### 3.8.2.1 Arguments to `km.analysis()`

* `input_mapping (Dict)`: Dictionary of mapping input data to a specific input. Input data should be a list of numpy array.
* `output_dir (str, optional)`: path to the output directory. Defaults to /data1/kneron_flow.
* `threads (int, optional)`: multithread setting. Defaults to 4.
* `quantize_mode (str, optional)`: quantize_mode setting. Currently support default and post_sigmoid. Defaults to "default".
* `datapath_range_method (str, optional)`: could be 'mmse' or 'percentage. mmse: use snr-based-range method. percentage: use arbitary percentage. Default to 'percentage'.
* `percentage (float, optional)`: used under 'percentage' mode. Suggest to set value between 0.999 and 1.0. Use 1.0 for detection models. Defaults to 0.999.
* `percentile (float, optional)`: used under 'mmse' mode. The range to search. The larger the value, the larger the search range, the better the performance but the longer the simulation time. Defaults to 0.001,
* `outlier_factor (float, optional)`: used under 'mmse' mode. The factor applied on outliers. For example, if clamping data is sensitive to your model, set outlier_factor to 2 or higher. Higher outlier_factor will reduce outlier removal by increasing range. Defaults to 1.0.
* `datapath_bitwidth_mode`: choose from "int8"/"int16". ("int16" not supported in kdp520).
* `weight_bitwidth_mode`: choose from "int8"/"int16". ("int16" not supported in kdp520).
* `model_in_bitwidth_mode`: choose from "int8"/"int16". ("int16" only for internal debug usage).
* `model_out_bitwidth_mode`: choose from "int8"/"int16". (currently should be same as model_in_bitwidth_mode).
* `fm_cut (str, optional)`: could be "default" or "deep_search". Deep search mode optimizes the performance but takes longer. Defaults to "default".
* `mode (int, optional)`: running mode for the analysis.
  * 0: run ip_evaluator only.
  * 1 (Defaults): run knerex (for quantization) only.
  * 2: run knerex + dynasty + compiler + csim + bit-true-match check. dynasty will inference only 1 image and only check quantization accuracy of output layers.
  * 3: run knerex + dynasty + compiler + csim + bit-true-match check. dynasty will inference all images and dump results of all layers. It will provide most detailed analysis but will take much longer time.
* `optimize (int, optional)`: level of optimization. 0-2, the larger number, the better model performance, but takes longer. Defaults to 0.
* `export_dynasty_dump (bool, optional)`: whether export the dump result when running dynasty. Defaults to False.

Returns: path to the output bie file. 

### 3.9 Fixed-Point Model Inference (Bie Model Check)

#### 3.9.1 Bie Model Generation

```python
input_image = Image.open('000000350003.jpg')
in_data = preprocess(input_image)
input_image.close()
out_data = ktc.kneron_inference([in_data], bie_file=bie_model_path, \
    input_names=['input_1_o0'], platform=520)
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)
```

#### 3.9.2 Bie Model Inference Result

```bash
(array([[260.754 ,471.40704,295.3402,522.4468],
[234.3568,210.12952,307.17825,389.8782]],dtype=float32),array([0.89978975,0.760541],dtype=float32),array([2,7],dtype=int32))
```

#### 3.9.3 The given `520` passed to ktc.kneron_inference() and passed to ktc.ModelConfig()

It's very important to note that 2 types of the given `520` passed to ktc.kneron_inference() VS. to ktc.ModelConfig() are different. 

* The given `520` type is INT for ktc.kneron_inference() to which the `520` is passed; and 
* The given `520` type is STRING for ktc.ModelConfig() to which the `520` is passed. 


### 3.10 Compilation

#### 3.10.1 Batch Compile

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

### 3.11 Hardware Simulation (Knef Model Ckeck)

#### 3.11.1 Knef Model Inference

```python
input_image = Image.open('000000350003.jpg')
in_data = preprocess(input_image)
input_image.close()
out_data = ktc.kneron_inference([in_data], nef_file=nef_model_path, \
    platform=int(DEVICE), input_names=[IMPUT_NAMES])
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)
```

#### 3.11.2 Knef Model Inference Result

```bash
(array([[260.754,471.40704,295.3402,522.4468],
[234.3568,210.12952,307.17825,389.8782]],dtype=float32),array([0.89978975,0.760541],dtype=float32),array([2,7],dtype=int32))
```

## 4. Deployment

First, we have to plug the `KL520` dongle into the `Notebook` USB port. 

### 4.1 Copy the Compiled Model from Docker

```bash
(base) root@48b04a35dc1a:/workspace/examples/darknet# \
    cp -fv /data1/kneron_flow/models_520.nef /docker_mount
```

After exit our own `Docker`, the copied Model, `models_520.nef`, can be found in `/mnt/docker` . 

### 4.2 Update The Model

```bash
cd C:\msys64\home\kneron_plus\res\models\KL520\tiny_yolo_v3
```

### 4.3 Run the given Example Routine

```bash
cd C:\msys64\home\kneron_plus\build\bin
```

```bash
kl520_demo_cam_generic_image_inference_drop_frame
```

### 4.4 kp_generic_image_inference_send() / kp_generic_image_inference_receive()

```C++
...
    int ret;
    kp_generic_image_inference_desc_t _input_data;
...
    _device = kp_connect_devices(1, &port_id, NULL);
    ret = kp_load_model_from_file(_device, _model_file_path, &_model_desc);
    ret = kp_inference_configure(_device, &infConf);
...
    _input_data.model_id = _model_desc.models[0].id;
    _input_data.inference_number = 0;
    _input_data.num_input_node_image = 1;
    _input_data.input_node_image_list[0].resize_mode = KP_RESIZE_ENABLE;
    _input_data.input_node_image_list[0].padding_mode = KP_PADDING_CORNER;
    _input_data.input_node_image_list[0].normalize_mode = KP_NORMALIZE_KNERON;
    _input_data.input_node_image_list[0].image_format = KP_IMAGE_FORMAT_RGB565;
    _input_data.input_node_image_list[0].width = _image_width;
    _input_data.input_node_image_list[0].height = _image_height;
    _input_data.input_node_image_list[0].crop_count = 0;
...
    for (;;;) {
        ret = kp_generic_image_inference_send(_device, &_input_data);
...
        ret = kp_generic_image_inference_receive(_device, \
                    &_output_desc, raw_output_buf, raw_buf_size);
...
    }
...
```


## Quiz

#### Q: Why choose `v0.23.0` instead of `latest`? 
A: Since we found the `AssertionError` happened for the given version, `v0.23.1`, which we display below: 

```bash
 Failure for model "input/input" when running "kdp520/compiler frontend"

===========================================
=            report on flow status        =
===========================================

                          kdp520                     general
               compiler frontend compiler_cfg onnx size (MB)
category case
input    input                 x            ✓             33

Traceback (most recent call last):
  File "compile.py", line 131, in <module>
    bie_model_path = km.analysis({IMPUT_NAMES: img_list}, output_dir='/data1/kneron_flow', threads=4, quantize_mode='default', datapath_range_method='percentage', fm_cut='deep_search', mode=1)
  File "/workspace/miniconda/lib/python3.7/site-packages/ktc/toolchain.py", line 166, in analysis
    export_dynasty_dump=export_dynasty_dump,
  File "/workspace/miniconda/lib/python3.7/site-packages/sys_flow/run.py", line 867, in gen_fx_model
    assert success, "Quantization model generation failed. See above message for details."
AssertionError: Quantization model generation failed. See above message for details.
```

#### Q: While coding python, can we use <TAB> to replace the leading space? 
A: Negative. 

#### Q: How to Convert YOLOv3 weights to Keres one? 

A: Take the following step: 

```bash
python /data1/keras_yolo3/convert.py `yolov3-tiny.cfg` `yolov3-tiny.weights` yolov3-tiny.h5
```

#### Q: While invoking a python function with a list of parameters, could we change the order of the given list? 
A: Yes, possitive. 

#### Q: Could we use ${PYTHONPATH} to replece `sys.path.append(str(pathlib.Path("/data1/keras_yolo3").resolve()))` described above?
A: Yes, possitive. 

#### Q: Why we set const `IMPUT_NAMES` to `'input_1_o0'` ? How the given `'input_1_o0'` to be determined? 
A: It's available to use the `Netron` app as the graph tool for the given ONNX model to check the model `INPUTS` name: 

![image](https://github.com/lexra/kpdocker/assets/33512027/ea6a46c9-b68b-42e5-8c58-4406d32e4d81)


#### Q: About `input_image = Image.open('000000350003.jpg')`, could we use a png file instead? 
A: Yes, possitive. 

#### Q: About `Workflow for Yolo Example` described above, is there any step that we could skip? If the answer is `possitive`, which step?

#### Q: The official documentation didn't invoke `ktc.convert_channel_last_to_first()` in `preprocess()`, Why we invoke `ktc.convert_channel_last_to_first()` before `preprocess()` returns? 

A: If we skip `ktc.convert_channel_last_to_first()`, the following error occurs: 

```bash
AssertionError:
        Input node (input_1_o0) has shape ((1, 3, 416, 416)),
        but the numpy list has different shapes of: [(416, 416, 3)].
        Please check the numpy input.
```

Example:

```python
import ktc
from PIL import Image
import numpy as np

image = Image.open("/workspace/examples/mobilenetv2/images/000007.jpg")
image = image.convert("RGB")
# Here the image is in channel last format, which is (224, 224, 3)
img_data = np.array(image.resize((224, 224), Image.BILINEAR)) / 255
ASSERT img_data.shape == (224, 224, 3)

# Now we use the API to convert the image into channel first format, which is (1, 3, 224, 224)
new_img_data = ktc.convert_channel_last_to_first(img_data)
ASSERT new_img_data.shape == (1, 3, 224, 224)
```







