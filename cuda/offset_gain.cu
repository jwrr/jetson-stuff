%%cu
// Add an offset and then apply a gain
// nvcc offset_gain.cu -o offset_gain

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Subtract offset and then muliply by (gain/64K).
__global__ void offset_gain(int* raw, int* offset, int* gain, int* corrected, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    corrected[tid] = (raw[tid] - offset[tid]) * gain[tid] / (64*1024);
  }
}

void offset_gain_cpu(int* raw, int* offset, int* gain, int* corrected, int N)
{
  for (int i = 0; i < N; i++) {
    corrected[i] = (raw[i] - offset[i]) * gain[i] / (64*1024);
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
  for (int i = r1; i < r1+width; i++) {
    for (int j = c1; j < c1+5; j++) {
      printf("[%d,%d]=%6d ", i, j, mat2d[i*n+j]);
    }
    printf("\n");
  }
 printf("\n");
}

// =====================================================================
// =====================================================================

int main()
{
  int ROWS = 480;
  int COLS = 640;
  int NNNN = ROWS * COLS;
  size_t bytes = NNNN * sizeof(int);

  int *raw, *offset, *gain, *corrected;
  cudaMallocManaged(&raw, bytes);
  cudaMallocManaged(&offset, bytes);
  cudaMallocManaged(&gain, bytes);
  cudaMallocManaged(&corrected, bytes);

  for (int i = 0; i < NNNN; i++) {
    raw[i] = rand() % 1024;
    offset[i] = rand() % 256 - 128;
    gain[i] = rand() % (64*1024);
  }

  int THREADS = 512;
  int BLOCKS = (NNNN + THREADS - 1) / THREADS;
  size_t SHMEM_SIZE = 0;
  startTimer();
  offset_gain<<<BLOCKS, THREADS, SHMEM_SIZE>>>(raw, offset, gain, corrected, NNNN);
  cudaDeviceSynchronize();
  stopTimer("gain_offset (GPU)"); 
  printSlice("Raw",       raw, COLS, 0, 0, 3, 3);
  printSlice("Offset",    offset, COLS, 0, 0, 3, 3);
  printSlice("Gain",      gain, COLS, 0, 0, 3, 3);
  printSlice("Corrected", corrected, COLS, 0, 0, 3, 3);
 
  startTimer();
  offset_gain_cpu(raw, offset, gain, corrected, NNNN);
  stopTimer("gain_offset (CPU)"); 
  printSlice("Corrected", corrected, COLS, 0, 0, 3, 3);
  return 0;
 }

