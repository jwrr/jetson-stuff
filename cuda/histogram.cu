/*
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin
%%cu
*/
// histogram.cu
// This program computes a histogram on the GPU using CUDA
// By: Nick from CoffeeBeforeArch. Retyped by jwrr.

// nvcc histogram.cu -o histogram

#include <cstdlib>
#include <iostream>
#include <time.h>

using namespace std;

// Define the kernel that will run on the GPU
__global__ void histogram(int* input, int* bins, int N_inputs, int N_bins, int DIV, int testMode)
{
  // Differentiate the threads. Every thread is going to bin one element
  // Calculate the global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ int s_bins[];
  // Shared Memory is shared within a thread. We have one thread per bin.
  // So, use the 1st N_bin threads to initialize the shared bins
  if (threadIdx.x < N_bins) {
      s_bins[threadIdx.x] = 0;
  }
  // Wait for all threads, which ensures the above initialization code has been
  // run.
   __syncthreads();

  // Boundary check
  if (tid < N_inputs) {
    int bin = input[tid] / DIV;
    if (testMode == 0) {
      // This leads to a race condition because all the threads are writing to
      // the bins array.
      bins[bin] += 1;
    } else if (testMode == 1) {
      // This avoids the above race condition
      atomicAdd(&bins[bin], 1);
    } else if (testMode == 2) {
      // now only the threads are competing against each other
      // so there are fewer performance reducing contention collisions.
      atomicAdd(&s_bins[bin], 1);
    }

  }

  // wait for all threads to finish updating their bins
  __syncthreads();

  if (threadIdx.x < N_bins) {
    atomicAdd(&bins[threadIdx.x], s_bins[threadIdx.x]);
  } 

}


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

// ==========================================================================
// ==========================================================================

int main()
{
  int N = 1 << 25;
  size_t bytes = N * sizeof(int);
  int N_bins = 10;
  size_t bytes_bins = N_bins * sizeof(int);
  int *input, *bins;

  // Use Unified Memory Model so explicit memory copies between host and device
  // are not needed.
  cudaMallocManaged(&input, bytes);
  cudaMallocManaged(&bins, bytes_bins);

  // Load input array with random data
  int MAX = 123;
  rand_array(input, N, MAX);

  int DIV = (100 + N_bins - 1) / N_bins;

  // Set the dimensions of our CTA (Cooperative Thread Array) and Grid
  // CTA is another name for Threadblock (or just Block) and is similar to an
  // OpenCL Workgroup.

  // A group of threads is called a CUDA block (CTA). CUDA blocks
  // are grouped into a grid. A KERNEL runs on a GRID of BLOCKS of THREADS.
  // A CUDA Block is executed on one streaming multiprocessor(SM).

  int THREADS = 512; // typically a value between 128 and 1024
  int BLOCKS = (N + THREADS - 1) / THREADS;
  size_t SHMEM_SIZE = N_bins * sizeof(int);

  for (int testMode = 0; testMode < 3; testMode++) {

    // Clear the histogram bins
    init_array(bins, N_bins, 0);

    double startTime;
    startTime = (double)clock()/CLOCKS_PER_SEC;
    // Call the kernel
    histogram<<<BLOCKS, THREADS, SHMEM_SIZE>>>(input, bins, N, N_bins, DIV, testMode);
    // Wait for all devices to finish
    cudaDeviceSynchronize();
    double endTime;
    double elapsedTime;
    endTime = (double)clock()/CLOCKS_PER_SEC;
    double elapsedTimeGPU = endTime - startTime;
    printf ("testMode = %d Elapsed time = %6.6f\n", testMode, elapsedTimeGPU);

    int tmp = 0;
    for (int i = 0; i < N_bins; i++){
      cout << i << ": " << bins[i] << endl;
      tmp += bins[i];
    }
    cout << tmp << endl;
  }

  cout << "done" << endl;
  return 0; 

} // main


