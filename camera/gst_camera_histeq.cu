// gst_camera.cpp
// MIT License
// Copyright (c) 2022 jwrr.com
// Inspired by:
// https://github.com/abubakr-shafique/Histogram_Equalization_CUDA_CPP and
// JetsonHacks

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;


// ============================================================================
// ============================================================================

__global__
void find_minmax_gpu(unsigned char* img, int N_CHAN, int* Min, int* Max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * N_CHAN;
  for (int i = 0; i < N_CHAN; i++) {
    atomicMin(&Min[i], img[pixel_id + i]);
    atomicMax(&Max[i], img[pixel_id + i]);
  }
}


__device__
int new_pixel_value(int Value, int Min, int Max)
{
  int Target_Min = 0;
  int Target_Max = 255;
  return (Target_Min + (Value - Min) * (int)((Target_Max - Target_Min)/(Max - Min)));
}


__global__
void histeq_gpu(unsigned char* img, int N_CHAN, int* Min, int* Max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * N_CHAN;
  for (int i = 0; i < N_CHAN; i++) {
    img[pixel_id + i] = new_pixel_value(img[pixel_id + i], Min[i], Max[i]);
  }
}

__global__
void make_low_contrast_for_testing(int n, uint8_t a, uint8_t *x, uint8_t *y)
{
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 if (i < n) y[i] = x[i] >> a;
}

// ============================================================================
// ============================================================================

void histeq_wrapper(unsigned char* img, int Height, int Width, int N_CHAN)
{
  unsigned char* Dev_Image = NULL;
  int* Dev_Min = NULL;
  int* Dev_Max = NULL;
  
  //allocate cuda variable memory
  cudaMalloc((void**)&Dev_Image, Height * Width * N_CHAN);
  cudaMalloc((void**)&Dev_Min, N_CHAN * sizeof(int));
  cudaMalloc((void**)&Dev_Max, N_CHAN * sizeof(int));
  
  int Min[3] = {255, 255, 255};
  int Max[3] = {0, 0, 0};
  
  //copy CPU data to GPU
  cudaMemcpy(Dev_Image, img, Height * Width * N_CHAN, cudaMemcpyHostToDevice);
  cudaMemcpy(Dev_Min, Min, N_CHAN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(Dev_Max, Max, N_CHAN * sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 Grid_Image(Width, Height);
  find_minmax_gpu <<<Grid_Image, 1>>>(Dev_Image, N_CHAN, Dev_Min, Dev_Max);
  histeq_gpu <<<Grid_Image, 1>>>(Dev_Image, N_CHAN, Dev_Min, Dev_Max);
  
  //copy memory back to CPU from GPU
  cudaMemcpy(img, Dev_Image, Height * Width * N_CHAN, cudaMemcpyDeviceToHost);
  
  //free up the memory of GPU
  cudaFree(Dev_Image);
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
        int N = vga_img.cols * vga_img.rows;
        uint8_t *d_x, *d_y;
        cudaMalloc(&d_x, N*sizeof(uint8_t));
        cudaMalloc(&d_y, N*sizeof(uint8_t));
        cudaMemcpy(d_x, vga_img.data, N*sizeof(uint8_t), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_y, vga_img.data, N*sizeof(uint8_t), cudaMemcpyHostToDevice);
        make_low_contrast_for_testing<<<(N+255)/256, 256>>>(N, 4, d_x, d_y);
        cudaMemcpy(low_contrast_img.data, d_y, N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaFree(d_x);
        cudaFree(d_y);

        histeq_wrapper(low_contrast_img.data, vga_img.rows, vga_img.cols, vga_img.channels());

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

