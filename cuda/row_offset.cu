!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin
%%cu

%%cu
// self_offset.cu
// This program applies a dynamic offset to a vector

// nvcc histogram.cu -o histogram

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
  const int MIDPOINT = 512;
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
      rowOffset /= NUM_CAL_VALS;
      sharedRow[0] = rowOffset - MIDPOINT;
    }
  }
  __syncthreads();

  // Subtract Offset
  if (tidValid) {
    value = imat[tid];
    bool applyOffset = threadIdx.x > NUM_CAL_VALS;
    if (applyOffset) {
      rowOffset = sharedRow[0];
      value -= rowOffset;
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
  const int MIDPOINT = 512;
  const  int NUM_CAL_VALS = 4;
  int value = 0;

  // Subtract Offset
  for (int r=0; r < 480; r++) {
    int rstart = r * N_COLS;
    int rowOffset = 0;
    for (int i = 0; i < NUM_CAL_VALS; i++) {
      rowOffset += imat[rstart+i];
    }
    rowOffset /= NUM_CAL_VALS;
    sharedRow[0] = rowOffset - MIDPOINT;

    for (int i=0; i < N_COLS; i++) {
      value = imat[i];
      bool applyOffset = i > NUM_CAL_VALS;
      if (applyOffset) {
        rowOffset = sharedRow[0];
        value -= rowOffset;
      }
      omat[rstart + i] = value;
    }
  }
} // rowOffsetKernel_CPU

// ============================================================================
// ============================================================================

void rand_array(int *a, int N, int MAX)
{
  for (int i = 0; i < N; i++) {
    a[i] = rand() % MAX;
  }
}

void init_array(int *a, int N, int value)
{
  for (int i = 0; i < N; i++) {
    a[i] = value;
  }
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

  int *imat, *omat;
  cudaMallocManaged(&imat, bytes);
  cudaMallocManaged(&omat, bytes);

  // Load imat array with random data
  int MAX = 123;
  rand_array(imat, NN, MAX);

  // Set the dimensions of our CTA (Cooperative Thread Array) and Grid
  // CTA is another name for Threadblock (or just Block) and is similar to an
  // OpenCL Workgroup.

  // A group of threads is called a CUDA block (CTA). CUDA blocks
  // are grouped into a grid. A KERNEL runs on a GRID of BLOCKS of THREADS.
  // A CUDA Block is executed on one streaming multiprocessor(SM).

  int THREADS = 512; // typically a value between 128 and 1024
  int BLOCKS = (NN + THREADS - 1) / THREADS;
  size_t SHMEM_SIZE = N_COLS * sizeof(int);

  for (int ii=0; ii < 5; ii++) {
    startTimer();
    rowOffsetKernel<<<BLOCKS, THREADS, SHMEM_SIZE>>>(imat, omat, NN, N_COLS);
    // Wait for all devices to finish
    cudaDeviceSynchronize();
    stopTimer("Row Offset (GPU)");
    printSlice("imat", imat, N_COLS, 0, 0, 8, 8);
    printSlice("omat", omat, N_COLS, 0, 0, 8, 8);

    startTimer();
    rowOffsetKernel_CPU(imat, omat, NN, N_COLS);
    // Wait for all devices to finish
    cudaDeviceSynchronize();
    stopTimer("Row Offset (CPU)");
    printSlice("omat", omat, N_COLS, 0, 0, 8, 8);
  } // for

  return 0;
} // main

