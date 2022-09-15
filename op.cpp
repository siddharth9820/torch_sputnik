#include "torch/script.h"
#include "torch/torch.h"
#include "sputnik/spmm/cuda_spmm.h"
#include <iostream>
#include "cuda_fp16.h"
#include "cuda.h"

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err; \
  } while (0)



torch::Tensor copy(torch::Tensor tensor){
	return tensor.clone();
}


//void BM_CudaSpmm_GenericFloat(benchmark::State& state) {
// const int kDimM = state.range(0);
//  const int kDimK = state.range(1);
//  const int kDimN = state.range(2);
//  const int kNonZeros = state.range(3);

//  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
//  absl::BitGen generator;
//  CudaSparseMatrix<half2> sparse_matrix_gpu(
//      kDimM, kDimK, kNonZeros, RANDOM_UNIFORM, &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
//  CudaMatrix<half2> matrix_gpu(kDimK, kDimN, &generator);

  // Create the output matrix on the gpu.
//  CudaMatrix<half2> output_matrix_gpu(kDimM, kDimN, &generator);

//  int batch_size = 10;
//  while (state.KeepRunningBatch(batch_size)) {
//    for (int i = 0; i < batch_size; ++i) {
//      CUDA_CALL(CudaSpmm(
//          kDimM, kDimK, kDimN, sparse_matrix_gpu.NumElementsWithPadding(),
//          sparse_matrix_gpu.RowIndices(), sparse_matrix_gpu.Values(),
//          sparse_matrix_gpu.RowOffsets(), sparse_matrix_gpu.ColumnIndices(),
//          matrix_gpu.Values(), output_matrix_gpu.Values(), 0));
//    }
//    CUDA_CALL(cudaStreamSynchronize(nullptr));
//  }
//  ReportThroughput(state);
//}


void spmm(const torch::Tensor& row_indices, 
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
	

	CUDA_CALL(sputnik::CudaSpmm(
	(int)m, (int)k, (int)n, (int)nonzeros,
	row_indices.data_ptr<int>(), (__half2*)values.data<torch::Half>(),
	row_offsets.data_ptr<int>(), (short2*)column_indices.data_ptr(),
	(__half2*)b.data<torch::Half>(), (__half2*)output.data<torch::Half>(), 0
	));
}



TORCH_LIBRARY(sputnik, m) {
  m.def("copy", copy);
  m.def("spmm", spmm);
}

