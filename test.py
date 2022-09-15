import torch
from sparse_utils import _dense_to_sparse, fp16_mask_correction
from sspade import extract_mask
torch.ops.load_library("build/libsputnik_ops.so")


#m ,k, n = 10, 10, 10
#nnz = 20;
#row_indices = torch.randint(low=0, high=m, size=(nnz,), dtype=torch.int32).cuda()
#column_indices = torch.randint(low=0, high=k, size=(nnz,), dtype=torch.int16).cuda()
#row_offsets = torch.randint(low=0, high=k, size=(m+1,), dtype=torch.int32).cuda()
#values = torch.randn(size=(nnz,), dtype=torch.float16, device='cuda')
#b = torch.randn(size=(k,n), dtype=torch.float16, device='cuda')
#output = torch.randn(size=(m,n), dtype=torch.float16, device='cuda')

fp16=False
sparsity = 90
in_dim = 128
out_dim = in_dim * 4
batch_size = 8

W_dense = torch.rand((out_dim, in_dim), device='cuda', dtype=torch.float32)

if fp16:
    mask = fp16_mask_correction(extract_mask(W_dense, sparsity))
    W_dense = W_dense * mask
    values, row_indices, row_offsets, column_indices =_dense_to_sparse(W_dense, 'cuda')
    values = values.type(torch.float16)
    row_indices = row_indices.type(torch.int32)
    column_indices = column_indices.type(torch.int16)
    b = torch.rand(size=(W_dense.shape[-1], batch_size), dtype=torch.float16, device='cuda')
    m = W_dense.shape[0]
    n = b.shape[-1]
    k = b.shape[0]
    output = torch.zeros(size=(m,n), dtype=torch.float16, device='cuda')

    torch.ops.sputnik.spmm_fp16(row_indices,
                    values,
                    row_offsets,
                    column_indices,
                    b,
                    output,
                    m,
                    n,
                    k
                    )

    W_dense = W_dense.type(torch.float16)
    torch.cuda.synchronize()
else:
    W_dense = W_dense.type(torch.float32)
    mask = extract_mask(W_dense, sparsity)
    W_dense = W_dense * mask
    values, row_indices, row_offsets, column_indices =_dense_to_sparse(W_dense, 'cuda')
    b = torch.rand(size=(W_dense.shape[-1], batch_size), dtype=torch.float32,
            device='cuda').contiguous()
    m = W_dense.shape[0]
    n = b.shape[-1]
    k = b.shape[0]
    output = torch.zeros(size=(m,n), dtype=torch.float32,
            device='cuda').contiguous()

    torch.cuda.synchronize()
    torch.ops.sputnik.spmm_fp32(row_indices,
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

out_dense= torch.matmul(W_dense, b)
print(f"FP16 = {fp16}, out_dim={out_dim}, in_dim={in_dim},batch_size={batch_size}, MSE={torch.mean((out_dense.float()-output.float())**2)}")
