%%cu
// image_correction.cu
// This program applies some offsets and a scaling factor.
// GPU appears to be ~1.5x faster than CPU.

// nvcc image_correction.cu -o image_corrections

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;

#define DATA_TYPE int

// ============================================================================
// ============================================================================

// Define the kernel that will run on the GPU
__global__ void imageCorrectionKernel(DATA_TYPE* imat, DATA_TYPE* offset, DATA_TYPE* gain, DATA_TYPE* omat, int NN, int N_COLS)
{
  extern __shared__ DATA_TYPE sharedRow[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool tidValid = tid < NN;
  const  int NUM_CAL_VALS = 4;
  DATA_TYPE value = 0;
  DATA_TYPE rowOffset = 0;

  // First element of row calculates rowOffset
  if (tidValid) {
    bool firstCol = threadIdx.x == 0;
    if (firstCol) {
      DATA_TYPE rowOffset = 0;
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
      value -= rowOffset;
      value = (value - offset[tid]) * gain[tid];
    }
    omat[tid] = value;
  }
  __syncthreads();

} // imageCorrectionKernel


// ============================================================================
// ============================================================================


// Define the kernel that will run on the GPU
void imageCorrectionKernel_CPU(DATA_TYPE* imat, DATA_TYPE* offset, DATA_TYPE* gain, DATA_TYPE* omat, int NN, int N_COLS)
{
  DATA_TYPE sharedRow[N_COLS];
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // bool tidValid = true; // tid < NN;
  const  int NUM_CAL_VALS = 4;
  DATA_TYPE value = 0;
  // Subtract Offset
  for (int r=0; r < 480; r++) {
    int rstart = r * N_COLS;
    DATA_TYPE rowOffset = 0;
    for (int i = 0; i < NUM_CAL_VALS; i++) {
      rowOffset += imat[rstart+i];
    }
    sharedRow[0] = rowOffset / NUM_CAL_VALS;

    rowOffset = sharedRow[0];
    for (int i=0; i < N_COLS; i++) {
      int tid = rstart + i;
      value = imat[tid];
      bool applyOffset = i >= NUM_CAL_VALS;
      if (applyOffset) {
        value -= rowOffset;
        value = (value - offset[tid]) * gain[tid];
      }
      omat[tid] = value;
    }
  }
} // imageCorrectionKernel_CPU


// ============================================================================
// ============================================================================


void randArray(DATA_TYPE *a, int N, int MIN, int MAX)
{
  int range = MAX - MIN + 1;
  for (int i = 0; i < N; i++) {
    a[i] = (DATA_TYPE)((rand() % range) + MIN);
  }
}


void initArray(DATA_TYPE *a, int N, DATA_TYPE value)
{
  for (int i = 0; i < N; i++) {
    a[i] = value;
  }
}


// return true if equal
bool eqArray(DATA_TYPE* a, DATA_TYPE* b, int N)
{
  for (int i = 0; i < N; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}



DATA_TYPE findMin(DATA_TYPE *a, int N)
{
  DATA_TYPE min = a[0];
  for (int i = 1; i < N; i++) {
    if (a[i] < min) {
      min = a[i];
    }
  }
  return min;
}


DATA_TYPE findMax(DATA_TYPE *a, int N)
{
  DATA_TYPE max = a[0];
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


void printSlice(char* title, DATA_TYPE* mat2d, int n, int r1, int c1, int height, int width)
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
  size_t bytes = NN * sizeof(DATA_TYPE);

  DATA_TYPE *imat, *omat_gpu, *omat_cpu;
  cudaMallocManaged(&imat, bytes);
  cudaMallocManaged(&omat_gpu, bytes);
  cudaMallocManaged(&omat_cpu, bytes);


  DATA_TYPE *offset, *gain;
  cudaMallocManaged(&offset, bytes);
  cudaMallocManaged(&gain, bytes);

  for (int i = 0; i < NN; i++) {
    offset[i] = rand() % 256 - 128;
    gain[i] = (DATA_TYPE)(rand() % 1024) / 256; // gain of 0 to 4
  }


  // Load imat array with random data
  int MIN = -128;
  int MAX = 127;

  int THREADS = 660; // typically a value between 128 and 1024
  int BLOCKS = (NN + THREADS - 1) / THREADS;
  size_t SHMEM_SIZE = N_COLS * sizeof(DATA_TYPE);

  // printSlice("imat", imat, N_COLS, 0, 0, 8, 8);

  for (int ii=0; ii < 10; ii++) {
    randArray(imat, NN, MIN, MAX);
    printf("imat min = %d, max = %d\n", findMin(imat, NN), findMax(imat, NN));

    startTimer();
    imageCorrectionKernel<<<BLOCKS, THREADS, SHMEM_SIZE>>>(imat, offset, gain, omat_gpu, NN, N_COLS);
    // Wait for all devices to finish
    cudaDeviceSynchronize();
    stopTimer("Row Offset (GPU)");
    // printSlice("imat", imat, N_COLS, 0, 0, 8, 8);
    // printSlice("omat", omat_gpu, N_COLS, 0, 0, 8, 8);

    startTimer();
    imageCorrectionKernel_CPU(imat, offset, gain, omat_cpu, NN, N_COLS);
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

