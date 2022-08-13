CUDA on Jetson Nano
===================

Setup
-----

The CUDA compiler (nvcc) should be preinstalled, but paths need to be set in
.bashrc.

```
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

Compile
-------

```
nvcc vector_add.cu -o vector_add.exe
nvcc matrix_mult.cu -o matrix_mult.exe
nvcc row_offset.cu -o row_offset.exe
nvcc row_offset_float.cu -o row_offset_float.exe
nvcc offset_gain.cu -o offset_gain.exe
nvcc image_correction.cu -o image_correction.exe
nvcc histogram.cu -o histogram.exe
nvcc convolution.cu -o convolution.exe
```

