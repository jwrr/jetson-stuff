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

WIFI on Jetson NANO
-------------------

* [Sparkfun Tutorial](https://learn.sparkfun.com/tutorials/adding-wifi-to-the-nvidia-jetson/all) - Good step-by-step.


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


