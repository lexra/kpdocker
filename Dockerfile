
###########################################################
# kneron/toolchain
###########################################################

FROM kneron/toolchain:v0.23.0

RUN apt update
RUN apt install -y vim p7zip-full p7zip-rar iputils-ping net-tools udhcpc cython rar libsqlite3-dev curl
RUN apt install -y dirmngr --install-recommends
RUN /workspace/miniconda/bin/pip install gdown==4.7.3

###########################################################
# cudnn
###########################################################
RUN systemctl set-default multi-user.target
#RUN systemctl set-default graphical.target

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN mkdir -p /etc/apt/sources.list.d && echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /' > /etc/apt/sources.list.d/cuda.list
RUN apt update
#RUN apt upgrade
RUN echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections && echo 'tzdata tzdata/Zones/Asia select Taipei' | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata locales
#RUN ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime
RUN DEBIAN_FRONTEND=noninteractive apt install -y cuda-11-0 libcudnn8 libcudnn8-dev libnccl2

###########################################################
# 
###########################################################
#RUN apt install -y dconf-editor
#RUN gsettings set org.gnome.desktop.media-handling automount false
#RUN gsettings set org.gnome.desktop.media-handling automount-open false

###########################################################
# tensorflow
###########################################################
#RUN /workspace/miniconda/bin/pip3 install tensorflow==1.14.0
#RUN /workspace/miniconda/bin/pip3 install tensorflow_datasets
#RUN /workspace/miniconda/bin/pip install pillow==8.1.2

###########################################################
# keras/backend/tensorflow_backend.py
###########################################################
RUN cd /data1 && git clone https://github.com/qqwweee/keras-yolo3.git keras_yolo3 && cd -
COPY keras/backend/tensorflow_backend.py /workspace/miniconda/lib/python3.7/site-packages/keras/backend
RUN git clone https://github.com/kneron/ONNX_Convertor.git

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
RUN cd examples/darknet && wget https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth && cd -

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

RUN mkdir -p examples/glove
COPY examples/glove/gloves.jpg examples/glove
COPY examples/glove/compile.py examples/glove
COPY examples/glove/test.txt examples/glove
RUN cd examples/glove && /workspace/miniconda/bin/gdown --id 1wr4WYg13Td18nOt9ufW-4YpbHHuLWrp4 && unzip -o datasets.zip && rm -rfv datasets.zip && cd -
RUN cd examples/glove && /workspace/miniconda/bin/gdown --id 1FQgOoqvUvzSGPbxRVL1Bgvl-0AUeHQ8N && cd -
COPY examples/glove/glove.cfg examples/glove
RUN cd examples/glove && cat glove.cfg | grep anchors | tail -1 | awk -F '=' '{print $2}' > glove.anchors && cd -
RUN cd examples/glove && /workspace/miniconda/bin/python /data1/keras_yolo3/convert.py ./glove.cfg ./glove.weights ./glove.h5 && cd -

###########################################################
# freihand2d
###########################################################
RUN rm -rfv /data1/voc_data50
RUN mkdir -p examples/freihand2d
COPY examples/freihand2d/latest_kneron_optimized.onnx examples/freihand2d
COPY examples/freihand2d/compile.py examples/freihand2d
RUN wget https://www.kneron.com/forum/uploads/112/SMZ3HLBK3DXJ.7z -O SMZ3HLBK3DXJ.7z && 7z x SMZ3HLBK3DXJ.7z -o/workspace/examples/freihand2d && rm -rfv SMZ3HLBK3DXJ.7z


###########################################################
# freihand2d
###########################################################
#RUN mkdir -p examples/crnn
#COPY examples/crnn/compile.py examples/crnn
#COPY examples/crnn/new_crnn.onnx examples/crnn
#COPY examples/crnn/new_crnn_cnn.onnx examples/crnn
#COPY examples/crnn/new_crnn_rnn.onnx examples/crnn
#COPY examples/crnn/crnn_vgg.onnx examples/crnn
#COPY examples/crnn/crnn_resnet.onnx examples/crnn
#COPY examples/crnn/demo_image.zip examples/crnn
#RUN cd examples/crnn && unzip demo_image.zip
