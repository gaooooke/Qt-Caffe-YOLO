TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

#system
INCLUDEPATH += /usr/local/lib \
               /usr/lib/x86_64-linux-gnu
LIBS += -L/usr/local/lib

#opencv
INCLUDEPATH += /home/gaokechen/Downloads/opencv-3.3.1/include \
               /home/gaokechen/Downloads/opencv-3.3.1/include/opencv \
               /home/gaokechen/Downloads/opencv-3.3.1/include/opencv2/
LIBS += -L /home/gaokechen/Downloads/opencv-3.3.1/build/lib/libopencv_*.so
#LIBS += -L /usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui

#caffe
INCLUDEPATH += /home/gaokechen/MobileNet-YOLO/include \
               /home/gaokechen/MobileNet-YOLO/build/src \
LIBS += -L/home/gaokechen/MobileNet-YOLO/build/lib/lib_*.so
LIBS += -L/home/gaokechen/MobileNet-YOLO/build/lib -lcaffe

#cuda cudnn
INCLUDEPATH += /usr/local/cuda/include
LIBS += -L/usr/local/cuda/lib64
LIBS += -lcudart -lcublas -lcurand

#caffe addition
LIBS += -lglog -lgflags -lprotobuf -lboost_system -lboost_thread -llmdb -lleveldb -lstdc++ -lcblas -latlas -lcudnn
