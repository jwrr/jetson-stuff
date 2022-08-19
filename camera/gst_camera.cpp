// gst_camera.cpp
// MIT License
// Copyright (c) 2022 jwrr.com
// Inspired by:
// Copyright (c) 2019-2022 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    int capture_width = 1280;
    int capture_height = 720;
    int display_width = 1280;
    int display_height = 720;
    int framerate = 30;
    int flip_method = 0;

/*
    string gst_in_string = 
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280," 
	"height=(int)720, framerate=(fraction)30/1 ! nvvidconv flip-method=0"
	" ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx"
	" ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
*/

    string gst_in_string = 
        "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=(int)1280,"
        "height=(int)720, framerate=(fraction)60/1 ! nvvidconv flip-method=0"
        " ! video/x-raw,width=(int)1280,height=(int)720,format=(string)BGRx"
        " ! videoconvert ! video/x-raw,format=(string)BGR ! appsink";

    cout << "gst_in_string: " << gst_in_string << endl;
    cv::VideoCapture gst_in(gst_in_string, cv::CAP_GSTREAMER);
    if (!gst_in.isOpened()) {
        cout << "Error - Failed to open gstreamer input" << endl;
        return -1;
    }

    string gst_out_string = "appsrc ! videoconvert ! ximagesink";
    cout << "gst_out_string: " << gst_out_string << endl;
    cv::VideoWriter gst_out;
    gst_out.open(gst_out_string, 0, (double)60, cv::Size(1280, 720), true);
    if (!gst_out.isOpened()) {
        cout << "Error - Failed to create gstreamer output" << endl;
        return -1;
    }

    cout << "Hit Ctrl-C to exit" << "\n";
    while (true) {
        cv::Mat img;
        if (!gst_in.read(img)) {
            cout << "Capture read error" << endl;
            break;
        }

        cv::Mat gray8_img;
        cv::cvtColor(img, gray8_img, cv::COLOR_BGR2GRAY);
        cv::Mat gray16_img;
        gray8_img.convertTo(gray16_img, CV_16U);

        // Process monochrome 16U
        // ...
        //

        cv::Mat gray8_img2;
        gray16_img.convertTo(gray8_img2, CV_8U);

        // ximagesink doesn't like GRAY8 so convert to BGR
        // GStreamer warning: cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 3.
        cv::Mat bgr_img;
	cv::cvtColor(gray8_img2, bgr_img, cv::COLOR_GRAY2BGR);
	
        gst_out.write(bgr_img);
        // gst_out << img;
    }

    gst_in.release();
    return 0;
}

