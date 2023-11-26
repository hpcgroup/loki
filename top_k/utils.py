import torch


def mask_top_k_elements_3d(tensor, k, dim=3):
    # For when k is more than the context length
    topk = min(k, tensor.shape[3])
    # Find the indices of the top k elements along the specified dimension
    _, indices = tensor.topk(topk, dim=dim, largest=True)

    # Create a mask with zeros and ones
    mask = torch.full_like(tensor, fill_value=float('-inf'))
    # mask.scatter_(dim, indices, 1)
    mask.scatter_(dim, indices, tensor.gather(dim, indices))

    # Apply the mask to the original tensor
    # masked_tensor = tensor * mask

    return mask
