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
        // dim checks
	TORCH_CHECK(row_indices.dim() == 1, "row indices matrix should be 1 dimensional");
	TORCH_CHECK(values.dim() == 1, "values matrix should be 1 dimensional");
	TORCH_CHECK(row_offsets.dim() == 1, "row offsets matrix should be 1 dimensional");
	TORCH_CHECK(column_indices.dim() == 1, "row indices matrix should be 1 dimensional");
	TORCH_CHECK(b.dim() == 2, "rhs matrix b should be 2 dimensional");
	TORCH_CHECK(output.dim() == 2, "output matrix should be 2 dimensional");

	// device checks
	TORCH_CHECK(row_indices.is_cuda(), "row indices matrix should be on gpu");
	TORCH_CHECK(values.is_cuda(), "values matrix should be on gpu");
	TORCH_CHECK(row_offsets.is_cuda(), "row offsets matrix should be on gpu");
	TORCH_CHECK(column_indices.is_cuda(), "column indices matrix should be on gpu");
	TORCH_CHECK(b.is_cuda(), "rhs matrix b should be on gpu");
	TORCH_CHECK(output.is_cuda(), "output matrix should be on gpu");

	// contiguous checks
	TORCH_CHECK(row_indices.is_contiguous(), "row indices matrix should be contiguous");
	TORCH_CHECK(values.is_contiguous(), "values matrix should be contiguous");
	TORCH_CHECK(row_offsets.is_contiguous(), "row offsets matrix should be contiguous");
	TORCH_CHECK(column_indices.is_contiguous(), "column indices matrix should be contiguous");
	TORCH_CHECK(b.is_contiguous(), "rhs matrix b should be contiguous");
	TORCH_CHECK(output.is_contiguous(), "output matrix should be contiguous");
	
	
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
	
        // dim checks
	TORCH_CHECK(row_indices.dim() == 1, "row indices matrix should be 1 dimensional");
	TORCH_CHECK(values.dim() == 1, "values matrix should be 1 dimensional");
	TORCH_CHECK(row_offsets.dim() == 1, "row offsets matrix should be 1 dimensional");
	TORCH_CHECK(column_indices.dim() == 1, "row indices matrix should be 1 dimensional");
	TORCH_CHECK(b.dim() == 2, "rhs matrix b should be 2 dimensional");
	TORCH_CHECK(output.dim() == 2, "output matrix should be 2 dimensional");

	// device checks
	TORCH_CHECK(row_indices.is_cuda(), "row indices matrix should be on gpu");
	TORCH_CHECK(values.is_cuda(), "values matrix should be on gpu");
	TORCH_CHECK(row_offsets.is_cuda(), "row offsets matrix should be on gpu");
	TORCH_CHECK(column_indices.is_cuda(), "column indices matrix should be on gpu");
	TORCH_CHECK(b.is_cuda(), "rhs matrix b should be on gpu");
	TORCH_CHECK(output.is_cuda(), "output matrix should be on gpu");

	// contiguous checks
	TORCH_CHECK(row_indices.is_contiguous(), "row indices matrix should be contiguous");
	TORCH_CHECK(values.is_contiguous(), "values matrix should be contiguous");
	TORCH_CHECK(row_offsets.is_contiguous(), "row offsets matrix should be contiguous");
	TORCH_CHECK(column_indices.is_contiguous(), "column indices matrix should be contiguous");
	TORCH_CHECK(b.is_contiguous(), "rhs matrix b should be contiguous");
	TORCH_CHECK(output.is_contiguous(), "output matrix should be contiguous");
	
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

