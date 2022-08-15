%%cu
// Convolution
// To run on colab:
//   !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
//   %load_ext nvcc_plugin
//   %%cu
// To compile from command line:
//   nvcc convolution.cu -o convolution

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATA_TYPE int

// ==========================================================================
// ==========================================================================
// Kernel

// Define the kernel that will run on the GPU
__global__ void conv_GPU(int* img, int* kern, int* result, int ROWS, int COLS, int KSIZE)
{
  extern __shared__ DATA_TYPE sharedRow[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int NN = ROWS * COLS;
  // if (tid < 10) printf("TID=%d, NN=%d\n", tid, NN);

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

void initArray(int *a, int N, int value)
{
  for (int i = 0; i < N; i++) {
    a[i] = value;
  }
}

void randArray(int *a, int N, int MIN, int MAX)
{
  int range = MAX - MIN + 1;
  for (int i = 0; i < N; i++) {
    a[i] = (rand() % range) + MIN;
  }
}

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


double stopTimer(char* title, bool printElapsedTime)
{
  double t2 = (double)clock()/CLOCKS_PER_SEC;
  double elapsedTime = t2 - t1;
  if (printElapsedTime) {
    printf ("%s: Elapsed time = %6.6f\n", title, elapsedTime);
  }
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
  int NK = KSIZE * KSIZE;
  int *kern;
  size_t kernBytes = KSIZE * sizeof(int);
  cudaMallocManaged(&kern, kernBytes);
  initArray(kern, NK, 0); // box filter
  kern[4] = 9;  // 0 1 2 3  #4#   5 6 7 8

  int MIN = -128;
  int MAX = 127;
  
  for (int ii = 0; ii < 10; ii++) {
  
    randArray(img,  NN, MIN, MAX);
    // randArray(kern, NK, 0, 4);
  
    //  printSlice("img", img, COLS, 10, 20, 5, 5);
  
    // ------------------------------------------------------------------------
    // Run on CPU
    startTimer();
    conv_CPU(img, kern, result_cpu, ROWS, COLS, KSIZE);
    double elapsedTimeCPU = stopTimer("CPU TIME", false);
    // printSlice("Result (CPU))", result_cpu, COLS, 10, 20, 5, 5); 
  
    // ------------------------------------------------------------------------
    // Accelerate with GPU
  
    int THREADS = COLS; // typically a value between 128 and 1024
    int BLOCKS = (NN + THREADS - 1) / THREADS;
    size_t SHMEM_SIZE = COLS * sizeof(int);
  
    startTimer();
    conv_GPU<<<BLOCKS, THREADS, SHMEM_SIZE>>>(img, kern, result_gpu, ROWS, COLS, KSIZE);
    cudaDeviceSynchronize();
    double elapsedTimeGPU = stopTimer("GPU TIME", false);
    //  printSlice("Result (GPU)", result_gpu, COLS, 10, 20, 5, 5);
    //  printf("GPU matmult done\n");
  
    // ------------------------------------------------------------------------
    // Results
   
    printf("%3d: ", ii);
    if (equalArray(result_gpu, result_cpu, NN)) {
      printf("PASS: GPU == CPU ");
    } else {
      printf("FAIL: GPU != CPU ");
    }
    
    double performanceIncrease =  elapsedTimeCPU / elapsedTimeGPU;
    printf("Performance Improvement = %3.2fx faster\n", performanceIncrease);

  } // for
  printf("Done\n");

  return 0;
}

