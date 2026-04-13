import torch


def pad_embeddings(embeddings):
    """
    embeddings: List[Tensor(O_i, D)]
    returns:
        padded: Tensor(B, Lmax, D)
        mask:   BoolTensor(B, Lmax)  True=valid, False=pad
        lengths: LongTensor(B,)
    """

    B = len(embeddings)
    device = embeddings[0].device
    dtype = embeddings[0].dtype
    D = embeddings[0].size(1)

    lengths = torch.tensor([e.size(0) for e in embeddings], device=device)
    max_len = int(lengths.max().item())

    padded = torch.zeros((B, max_len, D), device=device, dtype=dtype)
    mask = torch.zeros((B, max_len), device=device, dtype=torch.bool)

    for i, e in enumerate(embeddings):
        n = e.size(0)
        if n > 0:
            padded[i, :n] = e
            mask[i, :n] = True

    return padded, mask, lengths
