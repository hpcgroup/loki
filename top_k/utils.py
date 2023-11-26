import torch


def mask_top_k_elements_3d(tensor, k, dim=2):
    # Find the indices of the top k elements along the specified dimension
    if tensor.shape[dim] <= k:
        return tensor
    _, indices = tensor.topk(k, dim=dim, largest=True)

    # Create a mask with zeros and ones
    mask = torch.full_like(tensor, fill_value=float('-inf'))
    # mask.scatter_(dim, indices, 1)
    mask.scatter_(dim, indices, tensor.gather(dim, indices))

    # Apply the mask to the original tensor
    # masked_tensor = tensor * mask

    return mask


def test_mask():
    test_tensor = torch.rand(1, 1, 5, 5)
    print (test_tensor)
    ones = torch.ones_like(test_tensor, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    test_tensor[~tril_mask] = float('-inf')
    print (test_tensor)

    test_tensor = mask_top_k_elements_3d(test_tensor, 2, dim=3)
    print (test_tensor)

if __name__ == "__main__":
    test_mask()
