// gst_camera_histeq.cpp
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
void find_minmax_gpu(uint8_t* img, int n_chan, int* min, int* max)
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
  if (output_pixval > output_max) output_pixval = output_max;
  return output_pixval;
}


__global__
void linearstretch_gpu(uint8_t* img, int n_chan, int* min, int* max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * n_chan;
  for (int i = 0; i < n_chan; i++) {
    img[pixel_id + i] = new_pixel_value(img[pixel_id + i], min[i], max[i]);
  }
}


__global__
void histeq0_gpu(uint8_t* img, int n, uint8_t* hist)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int id = x + y * gridDim.x;
  // Init Histogram
  if (id < n && id < 256) {
    hist[id] = 0;
  }
}

  
__global__
void histeq1_gpu(uint8_t* img, int n, uint8_t* hist)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int id = x + y * gridDim.x;
  // Create Histogram
  if (id < n) {
    atomicAdd(hist[img[id]], 1); // FIXME: move hist to shared memory
  }
}

  
__global__
void histeq2_gpu(uint8_t* img, int n, uint8_t* hist)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int id = x + y * gridDim.x;
  // Convert Histogram to CDF
  if (id == 0) {
    for (int ii = 1; ii < 256; ii++) { // FIXME: convert to parallel
      hist[ii] += hist[ii-1];
    }
  }
}


__global__
void histeq3_gpu(uint8_t* img, int n, uint8_t* hist)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int id = x + y * gridDim.x;
  // Use Histogram for contrast enhanced output
  if (id < n) {
    img[id] = hist[img[id]];
  }
} // histeq_gpu


__global__
void make_low_contrast_for_testing_gpu(uint8_t *x, uint8_t a, int n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    uint8_t tmp = x[i] >> a;
    uint8_t offset = tmp * (a-1) / (2*a);
    x[i] = offset + tmp;
  }
};


// ============================================================================
// ============================================================================

void linearstretch_wrapper(uint8_t* img, int rows, int cols, int n_chan)
{
  uint8_t* d_img = NULL;
  int* d_min = NULL;
  int* d_max = NULL;
  
  //allocate cuda memory
  cudaMalloc((void**)&d_img, rows * cols * n_chan);
  cudaMalloc((void**)&d_min, n_chan * sizeof(int));
  cudaMalloc((void**)&d_max, n_chan * sizeof(int));
  
  int min[3] = {255, 255, 255};
  int max[3] = {0, 0, 0};
  
  //copy CPU data to GPU
  cudaMemcpy(d_img, img, rows * cols * n_chan, cudaMemcpyHostToDevice);
  cudaMemcpy(d_min, min, n_chan * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max, max, n_chan * sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 Grid_Image(cols, rows);
  find_minmax_gpu<<<Grid_Image, 1>>>(d_img, n_chan, d_min, d_max);
  linearstretch_gpu <<<Grid_Image, 1>>>(d_img, n_chan, d_min, d_max);
  
  //copy memory back to CPU from GPU
  cudaMemcpy(img, d_img, rows * cols * n_chan, cudaMemcpyDeviceToHost);
  
  //free up the memory of GPU
  cudaFree(d_img);
}


void histeq_wrapper(uint8_t* img, int rows, int cols)
{
  int N_IMG = rows * cols;
  int N_BYTES = N_IMG * sizeof(uint8_t);
  int N_HIST = 256;
  uint8_t *d_img, *d_hist;
  
  //allocate cuda memory
  cudaMalloc((void**)&d_img, N_BYTES);
  cudaMalloc((void**)&d_hist, N_HIST * sizeof(uint8_t));
  
  //copy CPU data to GPU
  cudaMemcpy(d_img, img, N_BYTES, cudaMemcpyHostToDevice);
  
  dim3 Grid_Image(cols, rows);

  // Steps: (0) init histogram, (1) make histogram, (2) make cdf, (3) enhance
  histeq0_gpu<<<Grid_Image, 1>>>(d_img, N_IMG, uint8_t* hist)
  cudaDeviceSynchronize();
  histeq1_gpu<<<Grid_Image, 1>>>(d_img, N_IMG, uint8_t* hist)
  cudaDeviceSynchronize();
  histeq2_gpu<<<Grid_Image, 1>>>(d_img, N_IMG, uint8_t* hist)
  cudaDeviceSynchronize();
  histeq3_gpu<<<Grid_Image, 1>>>(d_img, N_IMG, uint8_t* hist)
  cudaDeviceSynchronize();
  
  //copy GPU memory back to CPU memory
  cudaMemcpy(img, d_img, N_BYTES, cudaMemcpyDeviceToHost);
  
  //free up GPU memory
  cudaFree(d_img);
  cudaFree(d_hist);
} // histeq_wrapper


void make_low_contrast_for_testing_wrapper(cv::Mat img)
{
  int N = img.cols * img.rows;
  uint8_t *d_x;
  cudaMalloc(&d_x, N*sizeof(uint8_t));
  cudaMemcpy(d_x, img.data, N*sizeof(uint8_t), cudaMemcpyHostToDevice);
  make_low_contrast_for_testing_gpu<<<(N+255)/256, 256>>>(d_x, 4, N);
  cudaMemcpy(img.data, d_x, N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
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
//      linearstretch_wrapper(low_contrast_img.data, vga_img.rows, vga_img.cols, vga_img.channels());
        histeq_wrapper(low_contrast_img.data, vga_img.rows, vga_img.cols);

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

