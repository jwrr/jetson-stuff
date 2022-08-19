// This Program is based on Abubakr Shafique's (abubakr.shafique@gmail.com) program
// at https://github.com/abubakr-shafique/Histogram_Equalization_CUDA_CPP.
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

// ============================================================================
// ============================================================================

__global__ void find_minmax_gpu(unsigned char* img, int N_CHAN, int* Min, int* Max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * N_CHAN;
  for (int i = 0; i < N_CHAN; i++) {
    atomicMin(&Min[i], img[pixel_id + i]);
    atomicMax(&Max[i], img[pixel_id + i]);
  }
}


__device__ int new_pixel_value(int Value, int Min, int Max)
{
  int Target_Min = 0;
  int Target_Max = 255;
  return (Target_Min + (Value - Min) * (int)((Target_Max - Target_Min)/(Max - Min)));
}


__global__ void histeq_gpu(unsigned char* img, int N_CHAN, int* Min, int* Max)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int pixel_id = (x + y * gridDim.x) * N_CHAN;
  for (int i = 0; i < N_CHAN; i++) {
    img[pixel_id + i] = new_pixel_value(img[pixel_id + i], Min[i], Max[i]);
  }
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
  // image from Wikipedia: https://en.wikipedia.org/wiki/Histogram_equalization
  string img_path = "/home/jwrr/github/jetson-stuff/histogram_equalization/low_contrast.jpg";
  cv::Mat img_orig = cv::imread(img_path);
  cv::Mat img = cv::imread(img_path);
  if (img.empty()) {
    std::cout << "Error - Could not read image: " << img_path << std::endl;
    return 1;
  }
  cout << "Image Size: " << img.cols << "x" << img.rows << 
          ", Image Channels: " << img.channels() << endl;
  histeq_wrapper(img.data, img.rows, img.cols, img.channels());
  cv::imwrite("high_contrast.png", img);

  string windowName1 = "Low Contrast";
  cv::namedWindow(windowName1);
  cv::moveWindow(windowName1,100,100);
  cv::imshow(windowName1, img_orig);

  string windowName2 = "High Contrast";
  cv::namedWindow(windowName2);
  cv::moveWindow(windowName2,450,100);
  cv::imshow(windowName2, img);

  cv::waitKey(0); // Wait for any keystroke in the window
  cv::destroyWindow(windowName1);
  cv::destroyWindow(windowName2);
  return 0;
}

