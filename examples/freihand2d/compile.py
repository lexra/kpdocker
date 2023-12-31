import ktc
import os
from PIL import Image
import numpy as np


## Test Various converters.

## Clone https://github.com/kneron/ConvertorExamples first:
## git clone https://github.com/kneron/ConvertorExamples.git
## cd ConvertorExamples && git lfs pull

## Adjust loggin setting to avoid too many messages before start.
# import logging
# logging.basicConfig(level=logging.INFO)

## Keras to ONNX
# result_m = ktc.onnx_optimizer.keras2onnx_flow('/data1/ConvertorExamples/keras_example/onet-0.417197.hdf5')

## Pytorch to ONNX
# import torch
# import torch.onnx
# # Load the pth saved model
# pth_model = torch.load("/data1/ConvertorExamples/pytorch_example/resnet34.pth", map_location='cpu')
# # Export the model
# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(pth_model, dummy_input, '/data1/resnet34.onnx', opset_version=11)
# # Load the exported onnx model as an onnx object
# exported_m = onnx.load('/data1/resnet34.onnx')
# # Optimize the exported onnx object
# result_m = ktc.onnx_optimizer.torch_exported_onnx_flow(exported_m)

## Caffe to ONNX
# result_m = ktc.onnx_optimizer.caffe2onnx_flow('/data1/ConvertorExamples/caffe_example/mobilenetv2.prototxt', '/data1/ConvertorExamples/caffe_example/mobilenetv2.caffemodel')

## TF Lite to ONNX
# result_m = ktc.onnx_optimizer.tflite2onnx_flow('/data1/ConvertorExamples/tflite_example/model_unquant.tflite')

## ONNX Optimization
# optimized_m = ktc.onnx_optimizer.onnx2onnx_flow(result_m, eliminate_tail=True)

DEVICE = "720"

## Section 3
km = ktc.ModelConfig(32769, "8b28", DEVICE, onnx_path="/workspace/examples/freihand2d/latest_kneron_optimized.onnx")
eval_result = km.evaluate()

def preprocess(input_file):
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((224, 224), Image.BILINEAR)) / 255
    # The input data should be [C, H, W]
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    return img_data

input_data = [preprocess("/workspace/examples/freihand2d/voc_data50/2008_000992.jpg")]
inf_results = ktc.kneron_inference(input_data, onnx_file="/workspace/examples/freihand2d/latest_kneron_optimized.onnx", input_names=["input.1"])
if inf_results is not None:
    print('Section 3 E2E simulator finished.')
else:
    print('Section 3 E2E simulator failed.')
    exit(1)

## Section 4
# Preprocess images and create the input mapping
raw_images = os.listdir("/workspace/examples/freihand2d/voc_data50")
input_images = [preprocess("/workspace/examples/freihand2d/voc_data50/" + image_name) for image_name in raw_images]
input_mapping = {"input.1": input_images}

# Quantization
bie_path = km.analysis(input_mapping, threads = 4)

# E2E simulator (fixed point)
fixed_results = ktc.kneron_inference(input_data, bie_file=bie_path, input_names=["input.1"], platform=int(DEVICE))
if fixed_results is not None:
    print('Section 4 E2E simulator finished.')
else:
    print('Section 4 E2E simulator failed.')
    exit(1)

## Section 5
# Batch compile
compile_result = ktc.compile([km])

# E2E simulator (hardware)
hw_results = ktc.kneron_inference(input_data, nef_file=compile_result, platform=int(DEVICE), input_names=["input.1"])
if hw_results is not None:
    print('Section 5 E2E simulator finished.')
else:
    print('Section 5 E2E simulator failed.')
    exit(1)


try:
    np.testing.assert_almost_equal(fixed_results, hw_results, 4)
    print('Section 4 and Section 5 results are the same')
except Exception as mismatch:
    print("Section 4 and Section 5 results mismatch!")
    print(mismatch)
    exit(1)

print('')
exit(0)
