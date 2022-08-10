%%cu
// vector add
// nvcc -o vector_add.exe vector_add.cu

#include <stdio.h>
#include <stdlib.h>

__global__ void v_add(int* a, int* b, int* c, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if  (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

int main()
{
  int n = 1 << 20;

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

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  
  v_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  
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
  printf("Good Count = %d", good_count);
  
  return 0;
 }

