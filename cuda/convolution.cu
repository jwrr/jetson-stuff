%%cu
// Convolution
// To run on colab:
//   !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
//   %load_ext nvcc_plugin
//   %%cu
// To compile from command line:
//   nvcc convolutiont.cu -o convolution

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// ==========================================================================
// ==========================================================================
// Kernel

__global__ void conv(int* img, int* kern, int* result, int ROWS, int COLS, int KSIZE)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.x * blockDim.x;
  int col = threadIdx.x;
 
  int offset = KSIZE / 2; 
  int r = row - offset;
  int clow = col - offset;
  int sum = 0;
  for (int krow = 0; krow < KSIZE; krow++) {
    int c = clow;
    for (int kcol = 0; kcol < KSIZE; kcol++) {
      bool valid = (r >= 0 && c >= 0 && r < ROWS && c < COLS); 
      int val = valid ? img[r*COLS+c] : img[tid];
      sum += kern[krow*KSIZE+kcol] * val;
      c++;
    }
    r++;
  }
  result[tid] = sum / (KSIZE * KSIZE);
} // conv


void conv_CPU(int* img, int* kern, int* result, int ROWS, int COLS, int KSIZE)
{
//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
//  int row = blockIdx.x * blockDim.x;
//  int col = threadIdx.x;

  int offset = KSIZE / 2; 
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
  
      int tid = row*COLS + col; 
      int r = row - offset;
      int clow = col - offset;
      int sum = 0;
      for (int krow = 0; krow < KSIZE; krow++) {
        int c = clow;
        for (int kcol = 0; kcol < KSIZE; kcol++) {
          bool valid = (r >= 0 && c >= 0 && r < ROWS && c < COLS); 
          int val = valid ? img[r*COLS+c] : img[tid];
          sum += kern[krow*KSIZE+kcol] * val;
          c++;
        }
        r++;
      }
      result[tid] = sum / (KSIZE * KSIZE);

    }
  }

} // conv


// ==========================================================================
// ==========================================================================
// cuda_utils.cu


bool equalArray(int* a, int* b, int NN)
{
  for (int i = 0; i < NN; i++) {
    if (a[i] != b[i]) {return false;}
  }
  return true;
}


double t1;
void startTimer()
{
  t1 = (double)clock()/CLOCKS_PER_SEC;
}


double stopTimer(char* title)
{
  double t2 = (double)clock()/CLOCKS_PER_SEC;
  double elapsedTime = t2 - t1;
  printf ("%s: Elapsed time = %6.6f\n", title, elapsedTime);
 return elapsedTime;
}


void printSlice(char* title, int* mat2d, int n, int r1, int c1, int height, int width)
{
  printf("%s\n", title);
  for (int i = r1; i < r1+height; i++) {
    for (int j = c1; j < c1+width; j++) {
      printf("[%d,%d]=%6d ", i, j, mat2d[i*n+j]);
    }
    printf("\n");
  }
 printf("\n");
}


// ==========================================================================
// ==========================================================================
// Main

int main()
{
  int ROWS = 480;
  int COLS = 660;
  int NN = ROWS * COLS;
  size_t BYTES = NN * sizeof(int);

  int *img, *result_gpu, *result_cpu;
  cudaMallocManaged(&img, BYTES);
  cudaMallocManaged(&result_gpu, BYTES);
  cudaMallocManaged(&result_cpu, BYTES);

  int KSIZE = 3;
  int *kern;
  size_t kernBytes = KSIZE * sizeof(int);
  cudaMallocManaged(&kern, kernBytes);
  printf("ABC3\n");

  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
 //     img[i*COLS + j] = rand() % 1024; // 2; // rand() % 1024;
    }
  }
 
  for (int i = 0; i < NN; i++) {
    img[i] = rand() % 1024; // 2; // rand() % 1024;
  }
 
  printf("ABC4\n"); return 0;

  printSlice("img", img, NN, 10, 20, 5, 5);

  // ------------------------------------------------------------------------
  // Run on CPU
  startTimer();
  conv_CPU(img, kern, result_cpu, ROWS, COLS, KSIZE);
  double elapsedTimeCPU = stopTimer("CPU TIME");
  printSlice("Result (CPU))", result_cpu, NN, 10, 20, 5, 5); 

  // ------------------------------------------------------------------------
  // Accelerate with GPU

  int THREADS = 512; // typically a value between 128 and 1024
  int BLOCKS = (NN + THREADS - 1) / THREADS;
  size_t SHMEM_SIZE = COLS * sizeof(int);

  startTimer();
  conv<<<BLOCKS, THREADS, SHMEM_SIZE>>>(img, kern, result_gpu, ROWS, COLS, KSIZE);
  double elapsedTimeGPU = stopTimer("GPU TIME");
  printSlice("Result (GPU)", result_gpu, NN, 10, 20, 5, 5);
  printf("GPU matmult done\n");

  // ------------------------------------------------------------------------
  // Wrap up
  double performanceIncrease =  elapsedTimeCPU / elapsedTimeGPU;
  printf("Performance Improvement = %3.2fx faster\n\n", performanceIncrease);


  if (equalArray(result_gpu, result_cpu, NN)) {
    printf("PASS: GPU == CPU\n");
  } else {
    printf("FAIL: GPU != CPU\n");
  }
  printf("Done\n");

  return 0;
}

