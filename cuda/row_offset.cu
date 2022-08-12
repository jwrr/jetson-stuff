// !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
// %load_ext nvcc_plugin
// %%cu
// 
// // Jetson Nano Specs
// ====================
// 1 SM
// 4 Warps / SM (4 total)
// 32 Cuda Cores per Warp (128 cores total)
// 640MHz (Base Clock) to 920MHz (Boost Clock) 
// 4GB LPDDR4
// 
// // RTX 2080 Specs
// ====================
// 48 streaming multiprocessors (SM)
// 2 Warps / SM (96 total)
// 32 CUDA Cores per Warp (3072 cores total)
// 1.86 GHz
// 8GB GDDR6
// 
// %%cu
// row_offset.cu
// This program applies a dynamic offset to a vector

// nvcc row_offset.cu -o row_offsets

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;


// ============================================================================
// ============================================================================

// Define the kernel that will run on the GPU
__global__ void rowOffsetKernel(int* imat, int* omat, int NN, int N_COLS)
{
  extern __shared__ int sharedRow[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool tidValid = tid < NN;
  const  int NUM_CAL_VALS = 4;
  int value = 0;
  int rowOffset = 0;

  // First element of row calculates rowOffset
  if (tidValid) {
    bool firstCol = threadIdx.x == 0;
    if (firstCol) {
      int rowOffset = 0;
      int rowStart = tid;
      for (int i = 0; i < NUM_CAL_VALS; i++) {
        rowOffset += imat[rowStart+i];
      }
      sharedRow[0] = rowOffset / NUM_CAL_VALS;
    }
  }
  __syncthreads();

  // Subtract Offset
  if (tidValid) {
    value = imat[tid];
    bool applyOffset = threadIdx.x >= NUM_CAL_VALS;
    if (applyOffset) {
      rowOffset = sharedRow[0];
      value = rowOffset;
    }
    omat[tid] = value;
  }
  __syncthreads();

} // rowOffsetKernel


// ============================================================================
// ============================================================================


// Define the kernel that will run on the GPU
void rowOffsetKernel_CPU(int* imat, int* omat, int NN, int N_COLS)
{
  int sharedRow[N_COLS];
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // bool tidValid = true; // tid < NN;
  const  int NUM_CAL_VALS = 4;
  int value = 0;
  // Subtract Offset
  for (int r=0; r < 480; r++) {
    int rstart = r * N_COLS;
    int rowOffset = 0;
    for (int i = 0; i < NUM_CAL_VALS; i++) {
      rowOffset += imat[rstart+i];
    }
    sharedRow[0] = rowOffset / NUM_CAL_VALS;
    
    rowOffset = sharedRow[0];
    for (int i=0; i < N_COLS; i++) {
      value = imat[rstart + i];
      bool applyOffset = i >= NUM_CAL_VALS;
      if (applyOffset) {
        value = rowOffset;
      }
      omat[rstart + i] = value;
    }
  }
} // rowOffsetKernel_CPU


// ============================================================================
// ============================================================================


void randArray(int *a, int N, int MIN, int MAX)
{
  int range = MAX - MIN + 1;
  for (int i = 0; i < N; i++) {
    a[i] = (rand() % range) + MIN;
  }
}


void initArray(int *a, int N, int value)
{
  for (int i = 0; i < N; i++) {
    a[i] = value;
  }
}


// return true if equal
bool eqArray(int* a, int* b, int N)
{
  for (int i = 0; i < N; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
 


int findMin(int *a, int N)
{
  int min = a[0];
  for (int i = 1; i < N; i++) {
    if (a[i] < min) {
      min = a[i];
    }
  }
  return min;
}


int findMax(int *a, int N)
{
  int max = a[0];
  for (int i = 1; i < N; i++) {
    if (a[i] > max) {
      max = a[i];
    }
  }
  return max;
}


double t1;
void startTimer()
{
  t1 = (double)clock()/CLOCKS_PER_SEC;
}

void stopTimer(char* title)
{
  double t2 = (double)clock()/CLOCKS_PER_SEC;
  double elapsedTime = t2 - t1;
  printf ("%s: Elapsed time = %6.6f\n", title, elapsedTime);
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

int main()
{
  int N_ROWS = 480;
  int N_COLS = 660;
  int NN = N_ROWS * N_COLS;
  size_t bytes = NN * sizeof(int);

  int *imat, *omat_gpu, *omat_cpu;
  cudaMallocManaged(&imat, bytes);
  cudaMallocManaged(&omat_gpu, bytes);
  cudaMallocManaged(&omat_cpu, bytes);

  // Load imat array with random data
  int MIN = -128;
  int MAX = 127;

  int THREADS = 660; // typically a value between 128 and 1024
  int BLOCKS = (NN + THREADS - 1) / THREADS;
  size_t SHMEM_SIZE = N_COLS * sizeof(int);

  // printSlice("imat", imat, N_COLS, 0, 0, 8, 8);

  for (int ii=0; ii < 10; ii++) {
    randArray(imat, NN, MIN, MAX);
    printf("imat min = %d, max = %d\n", findMin(imat, NN), findMax(imat, NN));

    startTimer();
    rowOffsetKernel<<<BLOCKS, THREADS, SHMEM_SIZE>>>(imat, omat_gpu, NN, N_COLS);
    // Wait for all devices to finish
    cudaDeviceSynchronize();
    stopTimer("Row Offset (GPU)");
    // printSlice("imat", imat, N_COLS, 0, 0, 8, 8);
    // printSlice("omat", omat_gpu, N_COLS, 0, 0, 8, 8);

    startTimer();
    rowOffsetKernel_CPU(imat, omat_cpu, NN, N_COLS);
    stopTimer("Row Offset (CPU)");
    // printSlice("imat", imat, N_COLS, 0, 0, 8, 8);
    // printSlice("omat", omat_cpu, N_COLS, 0, 0, 8, 8);

    if (eqArray(omat_gpu, omat_cpu, NN)) {
        printf("PASS: GPU == CPU");
    } else {
        printf("FAIL: GPU != CPU");
    }

  } // for

  return 0;
} // main

