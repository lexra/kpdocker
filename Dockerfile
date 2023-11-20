
###########################################################
# kneron/toolchain
###########################################################

FROM kneron/toolchain:v0.23.0

RUN apt update
RUN apt install -y vim p7zip-full p7zip-rar iputils-ping net-tools udhcpc cython rar libsqlite3-dev
RUN apt install -y dirmngr --install-recommends
RUN /workspace/miniconda/bin/pip install gdown


###########################################################
# gcc-8 g++-8
###########################################################
#RUN apt install -y gcc-8 g++-8
#RUN ln -sf /usr/bin/g++-8 /usr/bin/g++
#RUN ln -sf /usr/bin/gcc-8 /usr/bin/gcc


###########################################################
# cudnn
###########################################################
RUN systemctl set-default multi-user.target

#RUN apt install -y dconf-editor
#RUN gsettings set org.gnome.desktop.media-handling automount false
#RUN gsettings set org.gnome.desktop.media-handling automount-open false

#RUN apt install -y nvidia-cuda-*
#RUN wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1810-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu1810-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
#RUN apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
#RUN apt update
#RUN apt install -y cuda-libraries-10-1 cuda-toolkit-10-1
#RUN ln -sf cuda-10.1 /usr/local/cuda
#RUN rm -rfv cuda-repo-ubuntu1810-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
#RUN /workspace/miniconda/bin/gdown --id 14CainePMLe_9da0p8R7seIjCXUc67j1j && tar zxvf cudnn-10.1-linux-x64-v7.6.5.32.tgz -C /usr/local && rm -rfv cudnn-10.1-linux-x64-v7.6.5.32.tgz
#RUN /workspace/miniconda/bin/gdown --id 1RzLiSid_2KWDzKFEHXxkWTDNVJ-qYH9q && dpkg -i nccl-repo-ubuntu1804-2.8.3-ga-cuda10.1_1-1_amd64.deb && rm -rfv nccl-repo-ubuntu1804-2.8.3-ga-cuda10.1_1-1_amd64.deb

#RUN systemctl set-default graphical.target


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
