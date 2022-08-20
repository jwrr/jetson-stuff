// gst_camera_linearstreth.cpp
// MIT License
// Copyright (c) 2022 jwrr.com
// Inspired by:
// JetsonHacks and others

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;


// ============================================================================
// ============================================================================


__global__
void find_minmax_gpu(unsigned char* img, int n_chan, int* min, int* max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * n_chan;
  for (int i = 0; i < n_chan; i++) {
    atomicMin(&min[i], img[pixel_id + i]);
    atomicMax(&max[i], img[pixel_id + i]);
  }
}


__device__
int new_pixel_value(int pixval, int min, int max)
{
  int output_min = 0;
  int output_max = 255;
  int output_rise = output_max - output_min;
  int input_run = max - min;
  int pixval_offset = pixval - min;
  int pixval_scaled = pixval_offset * output_rise / input_run;
  int output_pixval = output_min + pixval_scaled;
  if (output_pixval > output_max) then output_pixval = output_max;
  return output_pixval;
}


__global__
void linearstretch_gpu(unsigned char* img, int n_chan, int* min, int* max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * n_chan;
  for (int i = 0; i < n_chan; i++) {
    img[pixel_id + i] = new_pixel_value(img[pixel_id + i], min[i], max[i]);
  }
}


__global__
void make_low_contrast_for_testing_gpu(int n, uint8_t a, uint8_t *x)
{
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 if (i < n) x[i] = x[i] >> a;
}


// ============================================================================
// ============================================================================

void linearstretch_wrapper(unsigned char* img, int height, int width, int n_chan)
{
  unsigned char* d_img = NULL;
  int* d_min = NULL;
  int* d_max = NULL;
  
  //allocate cuda variable memory
  cudaMalloc((void**)&d_img, height * width * n_chan);
  cudaMalloc((void**)&d_min, n_chan * sizeof(int));
  cudaMalloc((void**)&d_max, n_chan * sizeof(int));
  
  int min[3] = {255, 255, 255};
  int max[3] = {0, 0, 0};
  
  //copy CPU data to GPU
  cudaMemcpy(d_img, img, height * width * n_chan, cudaMemcpyHostToDevice);
  cudaMemcpy(d_min, min, n_chan * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max, max, n_chan * sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 Grid_Image(width, height);
  find_minmax_gpu<<<Grid_Image, 1>>>(d_img, n_chan, d_min, d_max);
  linearstretch_gpu <<<Grid_Image, 1>>>(d_img, n_chan, d_min, d_max);
  
  //copy memory back to CPU from GPU
  cudaMemcpy(img, d_img, height * width * n_chan, cudaMemcpyDeviceToHost);
  
  //free up the memory of GPU
  cudaFree(d_img);
}


void make_low_contrast_for_testing_wrapper(cv::Mat img)
{
  int N = vga_img.cols * vga_img.rows;
  uint8_t *d_x;
  cudaMalloc(&d_x, N*sizeof(uint8_t));
  cudaMemcpy(d_x, vga_img.data, N*sizeof(uint8_t), cudaMemcpyHostToDevice);
  make_low_contrast_for_testing_gpu<<<(N+255)/256, 256>>>(N, 4, d_x);
  cudaMemcpy(low_contrast_img.data, d_y, N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
}


// ============================================================================
// ============================================================================

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
        "height=(int)720, framerate=(fraction)30/1 ! nvvidconv flip-method=0"
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
    gst_out.open(gst_out_string, 0, (double)30, cv::Size(640, 480), true);
    if (!gst_out.isOpened()) {
        cout << "Error - Failed to create gstreamer output" << endl;
        return -1;
    }

    cout << "Hit Ctrl-C to exit" << endl;
    while (cv::waitKey(1) == -1) {
        cv::Mat img;
        if (!gst_in.read(img)) {
            cout << "Capture read error" << endl;
            break;
        }

        cv::Mat gray8_img;
        cv::cvtColor(img, gray8_img, cv::COLOR_BGR2GRAY);
        //cv::Mat gray16_img;
        //gray8_img.convertTo(gray16_img, CV_16U);

        cv::Mat vga_img;
        cv::resize(gray8_img, vga_img, cv::Size(640, 480), cv::INTER_LINEAR);

        // Process monochrome
        // ...
        //

        cv::Mat low_contrast_img = vga_img.clone();
        make_low_contrast_for_testing_wrapper(low_contrast_img);
        linearstretch_wrapper(low_contrast_img.data, vga_img.rows, vga_img.cols, vga_img.channels());

        // cv::Mat gray8_img2;
        // gray16_img.convertTo(gray8_img2, CV_8U);

        // ximagesink doesn't like GRAY8 so convert to BGR
        // GStreamer warning: cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 3.
        cv::Mat bgr_img;
        // cv::cvtColor(gray8_img2, bgr_img, cv::COLOR_GRAY2BGR);
        cv::cvtColor(low_contrast_img, bgr_img, cv::COLOR_GRAY2BGR);
	
        gst_out.write(bgr_img);
        // gst_out << img;

    }

    gst_in.release();
    return 0;
}

