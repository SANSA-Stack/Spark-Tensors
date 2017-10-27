import torch


def kron(t1, t2):
    """
    Computes the Kronecker product between two matrix, row-wise.
    Adapted from: https://discuss.pytorch.org/t/kronecker-product/3919/5

    Params:
    -------
    t1, t2: Torch tensor of size M x N

    Returns:
    --------
    k: Torch tensor of size M x N^2
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(1, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, 1, t1_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2
