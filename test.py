import torch
torch.ops.load_library("build/libsputnik_ops.so")


m ,k, n = 10, 10, 10
nnz = 20;
row_indices = torch.randint(low=0, high=m, size=(nnz,), dtype=torch.int32).cuda()
column_indices = torch.randint(low=0, high=k, size=(nnz,), dtype=torch.int16).cuda()
row_offsets = torch.randint(low=0, high=k, size=(m,), dtype=torch.int32).cuda()
values = torch.randn(size=(nnz,), dtype=torch.float16, device='cuda')
b = torch.randn(size=(k,n), dtype=torch.float16, device='cuda')
output = torch.randn(size=(m,n), dtype=torch.float16, device='cuda')


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

W_dense = torch.zeros(m, k, dtype=torch.float16, device='cuda')
W_dense[row_indices.long(), column_indices.long()] = values
out_dense = torch.matmul(W_dense, b)

print(output)
print(out_dense)
