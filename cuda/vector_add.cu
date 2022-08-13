/*
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin
%%cu

%%cu
*/
// vector add
// nvcc vector_add.cu -o vector_add.exe

#include <stdio.h>
#include <stdlib.h>


__global__ void v_add(int* a, int* b, int* c, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if  (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}


void v_add_cpu(int* a, int* b, int* c, int n)
{
//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
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


int main()
{
  int n = 1 << 27;

  int* h_a;
  int* h_b;
  int* h_c;
  int* d_a;
  int* d_b;
  int* d_c;
  size_t bytes = n * sizeof(int);

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for (int i = 0; i < n; i++) {
    h_a[i] = 1; // rand() % 4096;
    h_b[i] = 2; // rand() % 4096;
  }

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  int block_size = 1024;
  int grid_size = (int)ceil( (float)n / block_size);
  printf("Grid size = %d\n", grid_size);

  startTimer();
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  v_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  stopTimer("Vector Add (GPU)");

  startTimer();
  v_add_cpu(h_a, h_b, h_c, n);
  stopTimer("Vector Add (CPU)");

  int good_count = 0;
  int bad_count = 0;
  for (int i = 0; i < n; i++) {
    if (h_c[i] != 3) {
      bad_count++;
      if (bad_count < 10) {
        printf("Error %d: c[%d] = %d\n", bad_count, i, h_c[i]);
      }
    } else {
      good_count++;
    }
  }
  printf("Good Count = %d\n", good_count);

  return 0;
}

