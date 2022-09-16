#include "torch/script.h"
#include "torch/torch.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/sddmm/cuda_sddmm.h"
#include <iostream>
#include "cuda_fp16.h"
#include "cuda.h"

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err; \
  } while (0)


void spmm_fp16(const torch::Tensor& row_indices, 
		const torch::Tensor& values,
		const torch::Tensor& row_offsets,
		const torch::Tensor& column_indices,
		const torch::Tensor& b,
		const torch::Tensor& output,
		int64_t m,
		int64_t n,
		int64_t k
		)
{
	TORCH_CHECK(b.dim() == 2);
	int nonzeros = column_indices.size(0);
	//TORCH_CHECK( nonzeros % 8 == 0, "non zeros need to be aligned to 32 bytes for vectorization")	

	CUDA_CALL(sputnik::CudaSpmm(
	(int)m, (int)k, (int)n, (int)nonzeros,
	row_indices.data_ptr<int>(), (__half2*)values.data<torch::Half>(),
	row_offsets.data_ptr<int>(), (short2*)column_indices.data_ptr(),
	(__half2*)b.data<torch::Half>(), (__half2*)output.data<torch::Half>(), 0
	));
}


void spmm_fp32(const torch::Tensor& row_indices, 
		const torch::Tensor& values,
		const torch::Tensor& row_offsets,
		const torch::Tensor& column_indices,
		const torch::Tensor& b,
		const torch::Tensor& output,
		int64_t m,
		int64_t n,
		int64_t k
		)
{
	TORCH_CHECK(b.dim() == 2);
	int nonzeros = column_indices.size(0);
	//TORCH_CHECK( nonzeros % 8 == 0, "non zeros need to be aligned to 32 bytes for vectorization")	

	CUDA_CALL(sputnik::CudaSpmm(
	(int)m, (int)k, (int)n, (int)nonzeros,
	row_indices.data_ptr<int>(), values.data_ptr<float>(),
	row_offsets.data_ptr<int>(),  column_indices.data_ptr<int>(),
	b.data_ptr<float>(), output.data_ptr<float>(), 0
	));
}


void sddmm_fp32(const torch::Tensor& row_indices, 
		const torch::Tensor& values,
		const torch::Tensor& row_offsets,
		const torch::Tensor& column_indices,
		const torch::Tensor& b, //rhs
		const torch::Tensor& output, //lhs
		int64_t m,
		int64_t n,
		int64_t k
		)
{
	
	TORCH_CHECK(b.dim() == 2);
	int nonzeros = column_indices.size(0);
      	CUDA_CALL(sputnik::CudaSddmm(
          (int)m, (int)n, (int)k, (int)nonzeros,
          row_indices.data_ptr<int>(),
          row_offsets.data_ptr<int>(),
          column_indices.data_ptr<int>(),
          output.data_ptr<float>(),
	  b.data_ptr<float>(),
          values.data_ptr<float>(), 0));
}

TORCH_LIBRARY(sputnik, m) {
  m.def("spmm_fp16", spmm_fp16);
  m.def("spmm_fp32", spmm_fp32);
  m.def("sddmm_fp32", sddmm_fp32);
}

