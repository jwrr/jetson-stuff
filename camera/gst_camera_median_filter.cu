// gst_camera_histeq.cpp_gpu
// MIT License
// Copyright (c) 2022 jwrr.com
// Inspired by:
// JetsonHacks and others

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

#define N_BLOCKS 2
#define N_THREADS_PER_BLOCK 640

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
  find_minmax_gpu<<<Grid_Image, N_THREADS_PER_BLOCK>>>(d_img, n_chan, d_min, d_max);
  linearstretch_gpu <<<Grid_Image, N_THREADS_PER_BLOCK>>>(d_img, n_chan, d_min, d_max);
  
  //copy memory back to CPU from GPU
  cudaMemcpy(img, d_img, rows * cols * n_chan, cudaMemcpyDeviceToHost);
  
  //free up the memory of GPU
  cudaFree(d_img);
}


// ============================================================================
// ============================================================================

__device__
void sort(int *a, int n, int stride)
{
  if(stride > 1) sort(a, n, stride / 2);
  for(int i=0; i < n-stride; i++){
    if(a[i] < a[i+stride]){
      // swap(a[i], a[i+stride]); // undefined in device code
      int tmp = a[i];
      a[i] = a[i+stride];
      a[i+stride] = tmp;
    }
  }
  if(stride > 1) sort(a, n, stride / 2);
}

// assumes n is a power of 2
__device__
void sort_array(int *a, int n)
{
  sort(a, n, n / 2);
}

// recursion causes this warning: ptxas warning : Stack size for entry 
// function '_Z17median_filter_gpuPhii' cannot be statically determined

__global__
void median_filter_gpu(uint8_t* img, int n_rows, int n_cols, const int n_filter_size)
{
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x;
  const int n_img = n_rows * n_cols;
  const int n_dim = n_filter_size;
  const int n_window = n_dim * n_dim;
  const int n_window_pow2 = 16; // 32;
  const int n_window_max = 32;
  const int n_half = n_dim / 2;
  for (int i = id; i < n_img; i += stride) {
    int window[n_window_max] = { 0 };
    int row = i / n_cols;
    int col = i % n_cols;
    
    // fill array to be sorted
    int window_i = 0;
    int row2 = row - n_half;
    for (int r = 0; r < n_dim; r++) {
      int col2 = col - n_half;
      for (int c = 0; c < n_dim; c++) {
        window[window_i++] = (int)img[row2 * n_cols + col2];
        col2++;
      }
      row2++;
    }
    
    sort_array(window, n_window_pow2);
    int median_value = window[n_window / 2];
    img[i] = (uint8_t)median_value;
    
  } // grid step
} // median_filter_gpu


void median_filter_wrapper(uint8_t* img, int n_rows, int n_cols, const int n_filter_size)
{
  int n_bytes = n_rows * n_cols * sizeof(uint8_t);
  uint8_t *d_img;
  cudaMalloc((void**)&d_img, n_bytes);
  cudaMemcpy(d_img, img, n_bytes, cudaMemcpyHostToDevice);
  median_filter_gpu<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_img, n_rows, n_cols, n_filter_size);
  cudaDeviceSynchronize();
  cudaMemcpy(img, d_img, n_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_img);
} // median_filter_wrapper


// ============================================================================
// ============================================================================



__global__
void add_popcorn_gpu(uint8_t* img, int n_rows, int n_cols)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x;
  const int n_img = n_rows * n_cols;
  for (int i = id; i < n_img; i += stride) {
    int add_noise = i % 1000;
    if (add_noise == 1) {
      img[i] = 254;
    }
  } // grid step
} // add_popcorn_gpu


void add_popcorn_wrapper(uint8_t* img, int n_rows, int n_cols)
{
  int n_bytes = n_rows * n_cols * sizeof(uint8_t);
  uint8_t *d_img;
  cudaMalloc((void**)&d_img, n_bytes);
  cudaMemcpy(d_img, img, n_bytes, cudaMemcpyHostToDevice);
  add_popcorn_gpu<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_img, n_rows, n_cols);
  cudaDeviceSynchronize();
  cudaMemcpy(img, d_img, n_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_img);
} // add_popcorn_wrapper


// ============================================================================
// ============================================================================


__global__
void histeq0_gpu(uint8_t* img, int n, int* hist)
{
  int id = threadIdx.x;
  int stride = blockDim.x;
  // Init Histogram
  if (id < n) {
    for (int i = id; i < 256; i += stride) {
      hist[i] = 0;
    }
  }
}


__global__
void histeq1_gpu(uint8_t* img, int n, int* hist)
{
  int id = threadIdx.x;
  int stride = blockDim.x;
  // Create Histogram
  for (int i = id; i < n; i += stride) {
    int bin = (int)img[i];
    atomicAdd(&hist[bin], 1); // FIXME: move hist to shared memory
  }
}

  
__global__
void histeq2_gpu(uint8_t* img, int n, int* hist)
{
  int id = threadIdx.x;
  // int stride = blockDim.x;
  // Convert Histogram to CDF
  if (id == 0) {
    for (int ii = 1; ii < 256; ii++) { // FIXME: convert to parallel
      hist[ii] += hist[ii-1];
    }
  }
}


__global__
void histeq3_gpu(uint8_t* img, int n, int* hist)
{
  int id = threadIdx.x;
  int stride = blockDim.x;
  // Use Histogram for contrast enhanced output
  for (int i = id; i < n; i += stride) {
    int h = hist[img[i]] * 256 / hist[255];
    if (h > 255) h = 255;
    img[i] = (uint8_t)h;
  }
} // histeq_gpu


void histeq_wrapper(uint8_t* img, int rows, int cols)
{
  int N_IMG = rows * cols;
  int N_BYTES = N_IMG * sizeof(uint8_t);
  int N_HIST = 256;
  uint8_t *d_img;
  int *d_hist;
  
  //allocate cuda memory
  cudaMalloc((void**)&d_img, N_BYTES);
  cudaMalloc((void**)&d_hist, N_HIST * sizeof(int));
  
  //copy CPU data to GPU
  cudaMemcpy(d_img, img, N_BYTES, cudaMemcpyHostToDevice);
  
  // Steps: (0) init histogram, (1) make histogram, (2) make cdf, (3) enhance
  histeq0_gpu<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_img, N_IMG, d_hist);
  cudaDeviceSynchronize();
  histeq1_gpu<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_img, N_IMG, d_hist);
  cudaDeviceSynchronize();
  histeq2_gpu<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_img, N_IMG, d_hist);
  cudaDeviceSynchronize();
  histeq3_gpu<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_img, N_IMG, d_hist);
  cudaDeviceSynchronize();
  
  //copy GPU memory back to CPU memory
  cudaMemcpy(img, d_img, N_BYTES, cudaMemcpyDeviceToHost);
  
  //free up GPU memory
  cudaFree(d_img);
  cudaFree(d_hist);
} // histeq_wrapper

// ============================================================================
// ============================================================================

__global__
void crop_gpu(uint8_t* img, int tr, int lc, int h, int w, int stride, int n)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int id = x + y * gridDim.x;
  // Use Histogram for contrast enhanced output
  if (id < n) {
    int br = tr + h - 1;
    int rc = lc + w - 1;
    if (y >= tr && y <= br && x >= lc && x <= rc) {
      int new_y = y - tr;
      int new_x = x - lc;
      int new_i = new_y * stride + new_x;
      img[new_i] = img[id];
    }
  }
} // crop_gpu


void crop_wrapper(cv::Mat img, int tr, int lc, int h, int w)
{
  int N = img.cols * img.rows;
  int N_BYTES = N*sizeof(uint8_t);
  int stride = img.cols;
  uint8_t *d_img;
  cudaMalloc(&d_img, N_BYTES);
  cudaMemcpy(d_img, img.data, N_BYTES, cudaMemcpyHostToDevice);
  crop_gpu<<<(N+255)/256, 256>>>(d_img, tr, lc, h, w, stride, N);
  cudaMemcpy(img.data, d_img, N_BYTES, cudaMemcpyDeviceToHost);
  cudaFree(d_img);
}

// ============================================================================
// ============================================================================

__global__
void make_low_contrast_for_testing_gpu(uint8_t *x, uint8_t a, int n)
{
  int id = threadIdx.x;
  int stride = blockDim.x;
  // Use Histogram for contrast enhanced output
  for (int i = id; i < n; i += stride) {
    uint8_t tmp = x[i] >> a;
    uint8_t offset = tmp * (a-1) / (2*a);
    x[i] = offset + tmp;
  }
};

void make_low_contrast_for_testing_wrapper(cv::Mat img)
{
  int N = img.cols * img.rows;
  uint8_t *d_x;
  cudaMalloc(&d_x, N*sizeof(uint8_t));
  cudaMemcpy(d_x, img.data, N*sizeof(uint8_t), cudaMemcpyHostToDevice);
  make_low_contrast_for_testing_gpu<<<1, N_THREADS_PER_BLOCK>>>(d_x, 4, N);
  cudaMemcpy(img.data, d_x, N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
}

// ============================================================================
// ============================================================================

// ============================================================================
// ============================================================================

int main()
{
    int disp_w = 640;
    int disp_h = 480;
    int frame_rate = 30;

    string gst_out_string = "appsrc ! videoconvert ! ximagesink";
    cout << "gst_out_string: " << gst_out_string << endl;
    cv::VideoWriter gst_out;
    gst_out.open(gst_out_string, 0, (double)frame_rate, cv::Size(disp_w, disp_h), true);
    if (!gst_out.isOpened()) {
        cout << "Error - Failed to create gstreamer output" << endl;
        return -1;
    }

/*
    string gst_in_string = 
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280," 
	"height=(int)720, framerate=(fraction)30/1 ! nvvidconv flip-method=0"
	" ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx"
	" ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
*/

    string gst_in_string = 
        "nvarguscamerasrc ! video/x-raw(memory:NVMM),"
        "width=1640,height=1232,framerate=" + to_string(frame_rate) + "/1"
        " ! nvvidconv ! video/x-raw,width=" + to_string(disp_w) + 
        ",height=" + to_string(disp_h) + ",format=BGRx"
        " ! videoconvert ! video/x-raw,format=BGR ! appsink";

    cout << "gst_in_string: " << gst_in_string << endl;
    cv::VideoCapture gst_in(gst_in_string, cv::CAP_GSTREAMER);
    if (!gst_in.isOpened()) {
        cout << "Error - Failed to open gstreamer input" << endl;
        return -1;
    }


    cout << "Hit Ctrl-C to exit" << endl;
    while (true) {
      int key = cv::waitKey(1);
      if (key != -1) {
        cout << "key=" << key << endl;
      }
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
      cv::resize(gray8_img, vga_img, cv::Size(disp_w, disp_h), cv::INTER_LINEAR);

      // Process monochrome
      // ...
      //

      const int n_filter_size = 3; // must be 3x3 or 5x5
      add_popcorn_wrapper(vga_img.data, vga_img.rows, vga_img.cols);
      median_filter_wrapper(vga_img.data, vga_img.rows, vga_img.cols, n_filter_size);

//      make_low_contrast_for_testing_wrapper(vga_img);
//    linearstretch_wrapper(low_contrast_img.data, vga_img.rows, vga_img.cols, vga_img.channels());
//      histeq_wrapper(vga_img.data, vga_img.rows, vga_img.cols);

      // cv::Mat gray8_img2;
      // gray16_img.convertTo(gray8_img2, CV_8U);

      // ximagesink doesn't like GRAY8 so convert to BGR
      // GStreamer warning: cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 3.
      cv::Mat bgr_img;
      // cv::cvtColor(gray8_img2, bgr_img, cv::COLOR_GRAY2BGR);
      cv::cvtColor(vga_img, bgr_img, cv::COLOR_GRAY2BGR);

      gst_out.write(bgr_img);
      // gst_out << img;

    }

    gst_in.release();
    return 0;
}
