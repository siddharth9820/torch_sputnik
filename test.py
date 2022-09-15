import torch
torch.ops.load_library("build/libsputnik_ops.so")


#m ,k, n = 10, 10, 10
#nnz = 20;
#row_indices = torch.randint(low=0, high=m, size=(nnz,), dtype=torch.int32).cuda()
#column_indices = torch.randint(low=0, high=k, size=(nnz,), dtype=torch.int16).cuda()
#row_offsets = torch.randint(low=0, high=k, size=(m+1,), dtype=torch.int32).cuda()
#values = torch.randn(size=(nnz,), dtype=torch.float16, device='cuda')
#b = torch.randn(size=(k,n), dtype=torch.float16, device='cuda')
#output = torch.randn(size=(m,n), dtype=torch.float16, device='cuda')

W_dense = torch.tensor([[0, 0, 1, 5], [10, 12, 0, 0], [0, 0, 0, 0], [1, 5, 0, 0]], dtype=torch.float16, device='cuda')
row_indices, column_indices = torch.nonzero(W_dense, as_tuple=True)
values = W_dense[row_indices, column_indices]
row_indices = row_indices.type(torch.int32)
column_indices = column_indices.type(torch.int16)
row_offsets = W_dense.to_sparse_csr().crow_indices().type(torch.int32)
b = torch.rand(size=(W_dense.shape[-1], 8), dtype=torch.float16, device='cuda')
m = W_dense.shape[0]
n = b.shape[-1]
k = b.shape[0]
output = torch.zeros(size=(m,n), dtype=torch.float16, device='cuda')

torch.ops.sputnik.spmm(row_indices,
		values,
		row_offsets,
		column_indices,
		b,
                output,
		m,
		n,
		k
		)
torch.cuda.synchronize()

out_dense = torch.matmul(W_dense, b)

print(output)
print(out_dense)
