// simple_camera.cpp
// MIT License
// Copyright (c) 2019-2022 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
               to_string(capture_height) + ", framerate=(fraction)" + to_string(framerate) + "/1" + 
           " ! nvvidconv flip-method=" + to_string(flip_method) + 
           " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
               to_string(display_height) + ", format=(string)BGRx" + 
           " ! videoconvert ! video/x-raw, format=(string)BGR" + 
           " ! appsink";
}


string matType(cv::Mat M)
{
  string r;
  uchar depth = M.type() & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (M.type() >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


string matDim(cv::Mat M)
{
    string r = to_string(M.cols) + "x" + to_string(M.rows);
    return r;
}

int main()
{
    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 30 ;
    int flip_method = 0 ;

    string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	cout<<"Failed to open camera."<<endl;
	return (-1);
    }

    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat rgb_img;
    cv::Mat gray8_img;
    cv::Mat gray16_img;

    cout << "Hit ESC to exit" << "\n" ;
    while(true)
    {
    	if (!cap.read(rgb_img)) {
		    cout<<"Capture read error"<<endl;
		    break;
	    }

        cv::cvtColor(rgb_img, gray8_img, cv::COLOR_BGR2GRAY);
        gray8_img.convertTo(gray16_img, CV_16UC1, 256, 0);
	cv::Mat blurred_img;
	//cv::GaussianBlur(gray16_img, blurred_img, cv::Size(11,11), 0);
	//cv::Mat hipass_img = gray16_img - blurred_img;
	//cv::imshow("MIPI-CSI Camera " + matDim(gray16_img) + " " + matType(gray16_img), hipass_img);
	cv::medianBlur(gray8_img, blurred_img, 11);
	cv::Mat hipass_img = gray8_img - blurred_img;
        cv::imshow("MIPI-CSI Camera " + matDim(gray8_img) + " " + matType(gray8_img), hipass_img); 
        // cv::imshow("MIPI-CSI Camera " + matDim(gray16_img) + " " + matType(gray16_img), blurred_img);
	// cv::imshow("MIPI-CSI Camera " + matDim(gray16_img) + " " + matType(gray16_img), gray16_img);

	    int keycode = cv::waitKey(10) & 0xff ; 
        if (keycode == 27) break ;
    }

    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}


