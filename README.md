# Torch-Sputnik - A PyTorch Interface for Optimized Sparse Matrix Multiplication Kernels For GPUs Using Sputnik

Sputnik is a library optimized for sparse GPU matrix multiplication for Neural Networks. This repo creates a PyTorch operators for the spMM and SDDMM kernels
present in Sputnik. 

## Build Instructions
[Pytorch](https://github.com/pytorch/pytorch) and [Sputnik](https://github.com/google-research/sputnik) should already be installed.

``mkdir build; cd build; cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" -DTORCH_CUDA_ARCH_LIST="7.0" -DSPUTNIK_DIR="/path/to/sputnik" ..``

## Issues
Sputnik fp16 has this weird constraint that number of non zeros in each row should be even.
