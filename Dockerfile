
#FROM kneron/toolchain:latest
FROM kneron/toolchain:v0.23.0
RUN apt update
RUN apt install -y vim p7zip-full p7zip-rar iputils-ping net-tools udhcpc cython rar
RUN /workspace/miniconda/bin/pip install gdown

#RUN apt-get nvidia-common
#RUN apt install -y nvidia-driver-535
#COPY epoch_100.pth /workspace/latest.pth
#RUN wget https://github.com/kneron/Model_Zoo/blob/main/mmpose/rsn18_freihand/latest.zip -O latest.zip

###########################################################
# tensorflow_backend.py
###########################################################
RUN cd /data1 && git clone https://github.com/qqwweee/keras-yolo3.git keras_yolo3 && cd -
COPY keras/backend/tensorflow_backend.py /workspace/miniconda/lib/python3.7/site-packages/keras/backend
#RUN /workspace/miniconda/bin/pip3 install pillow==8.1.2

###########################################################
# darknet
###########################################################
RUN mkdir -p examples/darknet
COPY examples/darknet/compile.py examples/darknet
COPY examples/darknet/yolov3-tiny.cfg examples/darknet
COPY examples/darknet/test_image10.txt examples/darknet
RUN cd examples/darknet && cat yolov3-tiny.cfg | grep anchors | tail -1 | awk -F '=' '{print $2}' > yolov3-tiny.anchors && cd -
RUN cd examples/darknet && wget https://pjreddie.com/media/files/yolov3-tiny.weights && cd -
RUN cd examples/darknet && wget http://doc.kneron.com/docs/toolchain/res/test_image10.zip && unzip test_image10.zip && cp /workspace/E2E_Simulator/app/test_image_folder/yolo/000000350003.jpg . && cd -
RUN cd examples/darknet && /workspace/miniconda/bin/python /data1/keras_yolo3/convert.py yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.h5 && cd -

###########################################################
# wheelchair
###########################################################
RUN mkdir -p examples/wheelchair
COPY examples/wheelchair/push_wheelchair.jpg examples/wheelchair
COPY examples/wheelchair/compile.py examples/wheelchair
COPY examples/wheelchair/test.txt examples/wheelchair
RUN cd examples/wheelchair && /workspace/miniconda/bin/gdown --id 1uSpN-bDlX9wG66K36yuscewB58pFnpbz && unzip -o datasets.zip && rm -rfv datasets.zip && cd -
RUN cd examples/wheelchair && /workspace/miniconda/bin/gdown --id 1K2fzXOUwuBjdBll3pHaldvqV41Rujsa_ && cd -
COPY examples/wheelchair/wheelchair.cfg examples/wheelchair
RUN cd examples/wheelchair && cat wheelchair.cfg | grep anchors | tail -1 | awk -F '=' '{print $2}' > wheelchair.anchors && cd -
RUN cd examples/wheelchair && /workspace/miniconda/bin/python /data1/keras_yolo3/convert.py ./wheelchair.cfg ./wheelchair.weights ./wheelchair.h5 && cd -

###########################################################
# freihand2d
###########################################################
RUN rm -rfv /data1/voc_data50
RUN mkdir -p examples/freihand2d
COPY examples/freihand2d/latest_kneron_optimized.onnx examples/freihand2d
COPY examples/freihand2d/compile.py examples/freihand2d
RUN wget https://www.kneron.com/forum/uploads/112/SMZ3HLBK3DXJ.7z -O SMZ3HLBK3DXJ.7z && 7z x SMZ3HLBK3DXJ.7z -o/workspace/examples/freihand2d && rm -rfv SMZ3HLBK3DXJ.7z
