import torch
from camtune.utils import DEVICE, DTYPE

device, dtype = DEVICE, DTYPE

def embedding_matrix(input_dim: int, target_dim: int) -> torch.Tensor:
    # return identity matrix if target size greater than input size
    if target_dim >= input_dim:  
        return torch.eye(input_dim, device=device, dtype=dtype)

    # add 1 to indices for padding column in matrix
    input_dims_perm = torch.randperm(input_dim, device=device) + 1

    # split dims into almost equally-sized bins and zero pad bins, the index 0 will be cut off later
    bins = torch.tensor_split(input_dims_perm, target_dim)  
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )  

    # add one extra column for padding
    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=dtype, device=device)  

    # fill mask with random +/- 1 at indices
    mtrx = mtrx.scatter_(
        1, bins, 
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )  

    return mtrx[:, 1:]  # cut off index zero as this corresponds to zero padding


def increase_embedding_and_observations(
    S: torch.Tensor, X: torch.Tensor, n_new_bins: int
) -> torch.Tensor:
    assert X.size(1) == S.size(0), "Observations don't lie in row space of S"

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].reshape(-1)

        if len(idxs_non_zero) <= 1:
            continue

        non_zero_elements = row[idxs_non_zero].reshape(-1)

        n_row_bins = min(
            n_new_bins, len(idxs_non_zero)
        )  # number of new bins is always less or equal than the contributing input dims in the row minus one

        new_bins = torch.tensor_split(idxs_non_zero, n_row_bins)[
            1:
        ]  # the dims in the first bin won't be moved
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        new_bins_padded = torch.nn.utils.rnn.pad_sequence(
            new_bins, batch_first=True
        )  # pad the tuples of bins with zeros to apply _scatter
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), device=device, dtype=dtype
        )  # submatrix to stack on S_update

        S_stack = S_stack.scatter_(
            1, new_bins_padded + 1, els_to_move_padded
        )  # fill with old values (add 1 to indices for padding column)

        S_update[
            row_idx, torch.hstack(new_bins)
        ] = 0  # set values that were move to zero in current row

        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)))
        )  # repeat observations for row at the end of X (column-wise)
        S_update = torch.vstack(
            (S_update, S_stack[:, 1:])
        )  # stack onto S_update except for padding column

    return S_update, X_update