!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin
%%cu

%%cu
// Matrix multiply
// To run on colab:
//   !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
//   %load_ext nvcc_plugin
//   %%cu
// To compile from command line:
//   nvcc matrix_mult.cu -o matrix_mult

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// ==========================================================================
// ==========================================================================
// Kernel

__global__ void matrixMul(int* m1, int* m2, int* p, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int p_sum = 0;
  for (int i = 0; i < n; i++) {
    p_sum += m1[row * n + i] * m2[i * n + column];
  }
  p[row * n + column] = p_sum;
}


void matrixMulSeq(int* m1, int* m2, int* p, int n)
{
  // int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int column = blockIdx.x * blockDim.x + threadIdx.x;
  for (int row = 0; row < n; row++) {

    for (int column = 0; column < n; column++) {
      int p_sum = 0;
      for (int i = 0; i < n; i++) {
        p_sum += m1[row * n + i] * m2[i * n + column];
      }
      p[row * n + column] = p_sum;
    }

  }
}


// ==========================================================================
// ==========================================================================
// cuda_utils.cu


bool cmpMat2d(int* m1, int* m2, int r, int c)
{
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (m1[i*c+j] != m2[i*c+j]) {return false;}
    }
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
  int n = 1 << 10;
  size_t bytes = n * n * sizeof(int);
  int* h_m1 = (int*)malloc(bytes);
  int* h_m2 = (int*)malloc(bytes);
  int* h_p = (int*)malloc(bytes);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      h_m1[i*n + j] = rand() % 1024; // 2; // rand() % 1024;
      h_m2[i*n + j] = rand() % 1024; //3; // rand() % 1024;
    }
  }

  printSlice("M1", h_m1, n, 10, 20, 5, 5);
  printSlice("M2", h_m2, n, 10, 20, 5, 5);

  // ------------------------------------------------------------------------
  // Run on CPU
  int* h_p_cpu = (int*)malloc(bytes);
  startTimer();
  matrixMulSeq(h_m1, h_m2, h_p_cpu, n);
  double elapsedTimeCPU = stopTimer("CPU TIME");
  printSlice("Product (CPU))", h_p_cpu, n, 10, 20, 5, 5); 

  // ------------------------------------------------------------------------
  // Accelerate with GPU
  startTimer();
  int* d_m1;
  int* d_m2;
  int* d_p;
  cudaMalloc(&d_m1, bytes);
  cudaMalloc(&d_m2, bytes);
  cudaMalloc(&d_p, bytes);
  cudaMemcpy(d_m1, h_m1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, h_m2, bytes, cudaMemcpyHostToDevice);
  int threads_per_block = 16;
  dim3 block_size (threads_per_block, threads_per_block);
  dim3 grid_size (n / block_size.x, n / block_size.y);
  matrixMul<<<grid_size, block_size >>>(d_m1, d_m2, d_p, n);
  cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
  double elapsedTimeGPU = stopTimer("GPU TIME");
  printSlice("Product (GPU)", h_p, n, 10, 20, 5, 5);
  printf("GPU matmult done\n");

  // ------------------------------------------------------------------------
 // Wrap up
  double performanceIncrease =  elapsedTimeCPU / elapsedTimeGPU;
  printf("Performance Improvement = %3.2fx faster\n\n", performanceIncrease);


  if (cmpMat2d(h_p, h_p_cpu, n, n)) {
    printf("PASS: GPU == CPU\n");
  } else {
    printf("FAIL: GPU != CPU\n");
  }
  printf("CPU matmult done\n");

  return 0;
}

