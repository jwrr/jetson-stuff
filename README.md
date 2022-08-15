JETSON NANO
===========

I don't know what I'm doing, but I'm having a good time.  This is my personal 
playground to try out my Jetson Nano and hopefully learn something.

Links
-----

* [Instructables]](https://www.instructables.com/Nvidia-Jetson-Nano-Tutorial-First-Look-With-AI-ML/)
* [Getting Started](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
* [User Guide](https://developer.nvidia.com/embedded/downloads#?search=Jetson%20Nano%20Developer%20Kit%20User%20Guide)
* [CUDA Crash Course by CoffeeBeforeArch on Youtube](https://www.youtube.com/watch?v=2NgpYFdsduY&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)


Day 1 - Unbox
-------------
Didn't get far. I found a micro-b cable, but need Ethernet cable and SD card.


Day 2 - Install
---------------
I scrounged an Ethernet cable and SD card.  Fortunately my laptop had an sd reader/writer.
Installation was easy and bring up was easy (just follow Getting Started). I'm writing
this post from the Jetson.


Day 3 - Run Demo
----------------
Run demos following Instructables Youtube video.

```
cd /usr/share/visionworks/sources
./install-samples.sh ~
cd ~
cd VisionWorks-1.6-Samples
make
cd ./bin/aarch64/linux/release
./nvx_demo_feature_tracker
./nvx_demo_hough_transform
  SPACE, M, ESC
```

Day 4 - Install OpenCV from Source
--------------------------
This takes a few hours.
 
* [Michael Degans' Github build script](https://github.com/mdegans/nano_build_opencv)
* [JetsonHacks OPENCV 4 + CUDA on Jetson Nano](https://jetsonhacks.com/2019/11/22/opencv-4-cuda-on-jetson-nano/)


```
$ git clone https://github.com/nano_build_opencv
$ cd nano_build_opencv
$ ./build_opencv.sh
WAIT A FEW HOURS...
$ opencv_version
4.4.0
$ which opencv_version
/usr/local/bin/opencv_version
$ls /usr/local/lib
libopencv_manyfiles.so
```


Day 5 - Python and OpenCV
-----------------

```
$ python --version
Python 2.7.17
$ python3 --version
Python 3.6.9

$python3
>>> import cv2
>>> print(cv2.getBuildInformation())
...
OpenCV modules:
    To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor ml objdetect optflow phase_unwrapping photo plot python2 python3 quality rapid reg rgbd saliency shape stereo stitching structured_light superres surface_matching text tracking video videoio videostab xfeatures2d ximgproc xobjdetect xphoto

...
  Other third-party libraries:
    Lapack:                      YES (/usr/lib/aarch64-linux-gnu/liblapack.so /usr/lib/aarch64-linux-gnu/libcblas.so /usr/lib/aarch64-linux-gnu/libatlas.so)
    Eigen:                       YES (ver 3.3.4)
    Custom HAL:                  YES (carotene (ver 0.0.1))
    Protobuf:                    build (3.5.1)

  NVIDIA CUDA:                   YES (ver 10.2, CUFFT CUBLAS FAST_MATH)
    NVIDIA GPU arch:             53 62 72
    NVIDIA PTX archs:

  cuDNN:                         YES (ver 8.0)

  OpenCL:                        YES (no extra features)

exit()
```

MIPI CSI Camera with OpenCV and Gstreamer
---------------------------

* [Jetson Nano + Raspberry Pi Camera by JetsonHacks](https://jetsonhacks.com/2019/04/02/jetson-nano-raspberry-pi-camera/), [github code](https://github.com/JetsonHacksNano/CSI-Camera)

* [GStreamer by TopTechBoy on Youtube](https://www.youtube.com/watch?v=_yU1kfcC6rY)

```
sudo apt install 44l-utils

## FIRST TRY WITH AUDIO
gst-launch-1.0 audiotestsrc ! alsasink
gst-inspect-1.0 audiotestsrc
gst-launch-1.0 audiotestsrc wave=1 freq=300 volume=1 ! alsasink
caps is capabilities
gst-inspect-1.0 alsasink
## Set caps for previous module (caps are separated by commas)
gst-launch-1.0 audiotestsrc wave=1 freq=300 volume=1 ! audio/x-raw,format=U8  ! alsasink

### The following fails because alsasink doesn't support U18LE so....
gst-launch-1.0 audiotestsrc wave=1 freq=300 volume=1 ! audio/x-raw,format=U18LE | audioconvert  ! alsasink
### So... add an audioconverter to the pipeline
gst-launch-1.0 audiotestsrc wave=1 freq=300 volume=1 ! audio/x-raw,format=U18LE | audioconvert  ! alsasink
### Usually you need to inspect the audio converters caps
gst-inspect-1.0 audioconvert to get input/output caps

gst-launch-1.0 audiotestsrc wave=1 freq=300 volume=1 ! audio/x-raw,format=U18LE | audioconvert ! audio/x-raw,format=U8 ! alsasink

## NOW TRY WITH VIDEO
gst-launch-1.0 videotestsrc ! ximagesink
gst-inspect-1.0 videotestsrc
gst-launch-1.0 videotestsrc pattern=11 ! ximagesink
gst-launch-1.0 videotestsrc pattern=0 ! video/x-raw,format=BGR  ! ximagesink
gst-inspect-1.0 ximagesink
gst-launch-1.0 videotestsrc pattern=0 ! video/x-raw,format=BGR  ! autovideoconvert  ! ximagesink
gst-launch-1.0 videotestsrc pattern=0 ! video/x-raw,format=BGR  ! autovideoconvert  ! videoconvert ! video/x-rww,width=1280,height=960  ! ximagesink
gst-inspect-1.0 videoconvert
gst-launch-1.0 videotestsrc pattern=0 ! video/x-raw,format=BGR  ! autovideoconvert  ! videoconvert ! video/x-rww,width=1280,height=960,framerate=30/1  ! ximagesink
gst-launch-1.0 videotestsrc pattern=0 ! video/x-raw,format=BGR  ! autovideoconvert  ! videoconvert ! video/x-rww,width=1280,height=960,framerate=1/2  ! ximagesink

gst-launch-1.0 nvarguscamerasrc ! autovideoconvert ! ximagesink
gst-launch-1.0 nvarguscamerasrc ! nvvidconv flip-method=2 ! video/x-raw,width=1280,height=840 ! autovideoconvert ! ximagesink
gst-launch-1.0 nvarguscamerasrc ! nvvidconv flip-method=2 ! video/x-raw,width=640,height=480 ! autovideoconvert ! ximagesink

gst-inspect-1.0 nvvidconv
gst-launch-1.0 nvarguscamerasrc ! video/x-raw  ! nvvidconv flip-method=2 ! video/x-raw,width=640,height=480 ! autovideoconvert ! ximagesink

gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3264,height=2464,framerate=21/1' ! nvvidconv flip-method=2 ! video/x-raw,width-640,height=480 !  ximagesink
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3264,height=2464,framerate=21/1' ! nvvidconv flip-method=2 ! video/x-raw,width-640,height=480 !  agingtv ! ximagesink
gst-inspect-1.0 agingtv
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3264,height=2464,framerate=21/1' ! nvvidconv flip-method=2 ! video/x-raw,width-640,height=480 !  agingtv scratchlines=20 ! ximagesink
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3264,height=2464,framerate=21/1' ! nvvidconv flip-method=2 ! video/x-raw,width-640,height=480 !  agingtv scratchlines=20 ! coloreffects preset=sepia ! ximagesink
















```


WIFI on Jetson Nano
-------------------

* [Sparkfun Tutorial](https://learn.sparkfun.com/tutorials/adding-wifi-to-the-nvidia-jetson/all) - Good step-by-step.
* It is very slow... needs debug, but going back to cat5 cable for now.

Arducam on Jetson Nano
----------------------

* Arducam MP IMX219 Camera Module
  * No software driver installation. Uses the Jetson Jetpack Nano camera driver.
* To roll back to the original Jetson Nano camera driver. This is only needed if you have a different driver installed.

```
sudo dpkg -r arducam-nvidia-l4t-kernel
```

* Check connection

```
ls /dev/video*
sudo apt install v4l-utils ### not needed. already installed
v4l2-ctl --list-formats-ext

``` 

* Run Camera
  * [Taking first picture with CSI](https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera)
```
nvgstcapture-1.0
  BOOM!!! VIDEO WORKS
  j<enter> captures an image
  q<enter> quit
  1<enter> to start recording hmmm... doesn't work
  0<enter> to stop recording

### Capture 10 seconds of video to file
nvgstcapture-1.0 --mode=2 --automate --capture-auto

IGNORE BELOW - NEEDS DEBUG
sudo apt install python3-pip
pip3 install v4l2-fix
git clone https://github.com/ArduCAM/MIPI_Camera.git
cd MIPI_Camera/Jetson/Jetvariety/example
python3 arducam_displayer.py -h
python3 arducam_displayer.py -d 0
  shows green screen?

sudo apt install vlc
```

* Use nVideo's video-viewer

```
git clone https://github.com/dusty-nv/jetson-utils.git
cd jetson-utils
mkdir builds
cd builds
cmake ..
make
cd ./aarch64/bin
./video-viewer csi://0
```

* To reboot or power off
```
sudo poweroff
sudo reboot
```

Canny Edge Detection (JETSONHACKS)
----------------------------------


Google Colab
------------

* File > New notebook
* Runtime > Change runtime type > GPU
* [Step by step. I started at step 5](https://www.geeksforgeeks.org/how-to-run-cuda-c-c-on-jupyter-notebook-in-google-colaboratory/)
  * Change "> > >" to ">>>" on line 48
* Ctrl-ML to turn on line numbers

```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin 
```


