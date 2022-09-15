import torch



def _diffsort(a):
    return torch.argsort(torch.diff(a), dim=0, descending=True)

def _nonzero_mask_to_sparse_csr_indices(mask, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(mask.shape) == 2
    index_dtype = torch.int32

    # Calculate the offset of each row.
    row_offsets = mask.sum(dim=-1, dtype=index_dtype).cumsum(dim=-1, dtype=index_dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))

    # Create the row indices and sort them.
    row_indices = _diffsort(row_offsets).to(index_dtype)

    # Extract the column indices for the nonzero values.
    column_indices = torch.where(mask)[1].to(index_dtype).contiguous()

    row_indices = row_indices.to(device)
    row_offsets = row_offsets.to(device)
    column_indices = column_indices.to(device)
    return row_indices, row_offsets, column_indices

def _dense_to_sparse(matrix, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(matrix.shape) == 2
    value_dtype = torch.float32

    # Extract the nonzero values.
    mask = matrix != 0
    values = matrix[mask].to(dtype=value_dtype, device=device)

    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(
        mask, device
    )
    return values, row_indices, row_offsets, column_indices


def fp16_mask_correction(mask):
    non_zeros_each_row = mask.sum(dim=1)
    for i in range(mask.shape[0]):
        if int(non_zeros_each_row[i]) % 2 != 0:
            for j in range(mask.shape[1]):
                if int(mask[i][j]) == 0:
                    mask[i][j] = 1
                    break
    non_zeros_each_row = mask.sum(dim=1)
    return mask
