import ktc
import os
import onnx
from PIL import Image
import numpy as np
import re

###  post process function  ###
import tensorflow as tf
import pathlib
import sys
sys.path.append(str(pathlib.Path("/data1/keras_yolo3").resolve()))
from yolo3.model import yolo_eval

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

###########################################################
CWD = '/workspace/examples/darknet/'
NAME = 'yolov3-tiny'
IN_MODEL_PREPROCESS = True
TEST_LIST = 'test_image10.txt'
IMPUT_NAMES = 'input_1_o0'
TEST_PICTURE = '000000350003.jpg'

DEVICE = '630'
if len(sys.argv) > 1:
    DEVICE = sys.argv[1]

CLASSES = int(darknetKeyValue(CWD + NAME + '.cfg', key='classes'))
CHANNELS = int(darknetKeyValue(CWD + NAME + '.cfg', key='channels'))
WIDTH = int(darknetKeyValue(CWD + NAME + '.cfg', key='width'))
HEIGHT = int(darknetKeyValue(CWD + NAME + '.cfg', key='height'))

def postprocess(inf_results, ori_image_shape):
    tensor_data = [tf.convert_to_tensor(data, dtype=tf.float32) for data in inf_results]
    # get anchor info
    anchors_path = CWD + NAME + '.anchors'
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    # post process
    num_classes = CLASSES
    boxes, scores, classes = yolo_eval(tensor_data, anchors, num_classes, ori_image_shape)
    with tf.compat.v1.Session() as sess:
        boxes = boxes.eval()
        scores = scores.eval()
        classes = classes.eval()
    return boxes, scores, classes

###  pre process function  ###
from yolo3.utils import letterbox_image

def preprocess(pil_img):
    model_input_size = (WIDTH, HEIGHT)  # to match our model input size when converting
    boxed_image = letterbox_image(pil_img, model_input_size)
    np_data = np.array(boxed_image, dtype='float32')
    # change normalization method due to we add "pixel_modify" BN node at model's front
    #np_data /= 255.
    if IN_MODEL_PREPROCESS == True:
        np_data -= 128
    else: 
        np_data /= 255.
    np_data = ktc.convert_channel_last_to_first(np_data)
    return np_data

# convert h5 model to onnx
print(CWD + NAME + '.h5')
m = ktc.onnx_optimizer.keras2onnx_flow(CWD + NAME + '.h5', optimize=0, input_shape=[1,WIDTH,HEIGHT,CHANNELS])
if DEVICE == '720':
    m = ktc.onnx_optimizer.onnx2onnx_flow(m, disable_fuse_bn=True, bgr=False, norm=False, rgba2yynn=False, eliminate_tail=True, opt_matmul=True, opt_720=True, duplicate_shared_weights=False)
else:
    m = ktc.onnx_optimizer.onnx2onnx_flow(m, disable_fuse_bn=True, bgr=False, norm=False, rgba2yynn=False, eliminate_tail=True, opt_matmul=True, opt_720=False, duplicate_shared_weights=False)

# add pixel modify node:
#   1. scaling 1/255 for every channel due to original normalize method, 
#   2. shift 0.5 to change input range from 0~255 to -128 to 127
if IN_MODEL_PREPROCESS == True:
    ktc.onnx_optimizer.pixel_modify(m, [1/255,1/255,1/255], [0.5,0.5,0.5])

# do onnx2onnx again to calculate "pixel_modify" BN node's output shape
m = ktc.onnx_optimizer.onnx2onnx_flow(m)

print(CWD + NAME + '.opt.onnx')
onnx.save(m, CWD + NAME + '.opt.onnx')

# setup ktc config
km = ktc.ModelConfig(205, "0001", DEVICE, onnx_model=m)

# npu(only) performance simulation
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + str(eval_result))

## onnx model check
input_image = Image.open(CWD + TEST_PICTURE)
in_data = preprocess(input_image)
input_image.close()
out_data = ktc.kneron_inference([in_data], onnx_file=CWD + NAME + '.opt.onnx', input_names=[IMPUT_NAMES])
if out_data is not None:
    print('E2E simulator finished.')
else:
    print('E2E simulator failed.')
    exit(1)
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)

# load and normalize all image data from folder
print('')
img_list = []
with open (CWD + TEST_LIST, "r") as myfile:
    lines = myfile.read().splitlines()
for item in lines:
    print(item)
    image = Image.open(item)
    img_data = preprocess(image)
    img_list.append(img_data)
    image.close()

# fix point analysis
bie_model_path = km.analysis({IMPUT_NAMES: img_list}, output_dir='/data1/kneron_flow', threads=4, quantize_mode='default', datapath_range_method='percentage',, mode=1)
print("\nFix point analysis done. Save bie model to '" + str(bie_model_path) + "'")

#if DEVICE != '520':
# bie model check
input_image = Image.open(CWD + TEST_PICTURE)
in_data = preprocess(input_image)
input_image.close()
#radix = ktc.get_radix(img_list)
out_data = ktc.kneron_inference([in_data], bie_file=bie_model_path, input_names=[IMPUT_NAMES], platform=int(DEVICE))
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)

# compile
print('')
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")

# nef model check
print('')
input_image = Image.open(CWD + TEST_PICTURE)
in_data = preprocess(input_image)
input_image.close()
#radix = ktc.get_radix(img_list)
out_data = ktc.kneron_inference([in_data], nef_file=nef_model_path, platform=int(DEVICE), input_names=[IMPUT_NAMES])
det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]])
print(det_res)

print('')

# /workspace/libs/compiler/kneron_nef_utils --extract /data1/kneron_flow/models_520.nef --keep_all
# /workspace/libs/compiler/kneron_nef_utils --info /data1/kneron_flow/models_520.nef
# /workspace/libs/compiler/kneron_nef_utils --combine_nef /data1/kneron_flow/models_520.nef

exit(0)
