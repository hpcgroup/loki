# This plot scores the PCA transforms on different datasets
# TODO: Remove this file
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from parse import parse
import numpy as np


def get_file_idx(file_name, tensor_type='key'):
    file_template = "tensor_key_{:d}_{:d}.pt"
    file_template = file_template.replace("key", tensor_type)
    file_name = file_name.split("/")[-1]
    config = parse(file_template, file_name)
    if config is None:
        print (f"[ERROR] Incorrect filename format: {file_name}")
        sys.exit(1)

    _, idx = config.fixed
    return idx

def load_tensors_for_layer(layer_id, folder_path, tensor_type='key'):
    # Find all files matching the pattern
    files = glob.glob(os.path.join(folder_path, f"tensor_{tensor_type}_{layer_id}_*.pt"))


    files = sorted(files, key=lambda x: get_file_idx(x, tensor_type))
    #print (files)

    if not files:
        print(f"No files found for layer_id {layer_id}")
        return None

    tensors = []
    # Load tensors from each file
    for file in files:
        try:
            tensor = torch.load(file)
            tensors.append(tensor)
        except Exception as e:
            print(f"Error loading tensor from {file}: {e}")

    return tensors


def compute_pca(layer_id, head, tensors, folder_path, tensor_type='key'):
    #tensors = load_tensors_for_layer(layer_id, folder_path, tensor_type)

    #Change the list of tensors to a tensor of shape (num_tensors, batch_size, num_heads, num_tokens, num_keys)
    #tensors = torch.stack(tensors[:-1], dim=0)

    tensors = tensors[:,:,head,:,:]
    #print (tensors.shape)

    tensors = tensors.reshape(-1, tensors.shape[-1])
    #print (tensors.shape)

    # Compute the PCA of the key vectors
    pca = PCA()
    pca.fit(tensors.cpu().numpy())

    return pca

    print (pca.components_.shape)

    # Plot the eigenvalues
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Principal component index')
    plt.ylabel('Explained variance')
    plt.title('Explained variance per principal component')

    plt.show()
    plt.savefig('pca.png')
  

def main():
    # I want to compute the PCA of all layers and all the heads
    tensor_type = sys.argv[1]
    num_layers = int(sys.argv[2])
    num_heads = int(sys.argv[3])
    hidden_size = int(sys.argv[4])
    folder_path = sys.argv[5] 
    rotary_type = ['prerotary', 'postrotary']

    datasets = os.listdir(folder_path)

    for rtype in rotary_type:
        for i, dataset1 in enumerate(datasets):
          dataset1_path = f"{folder_path}/{dataset1}/{rtype}/{tensor_type}"
          for layer_id in range(num_layers):
              print (f"Computing for layer {layer_id}, dataset1: {dataset1}, rotary: {rtype}")
              print (dataset1_path)
              # Tensor per layer of shape (num_heads, head_dim, head_dim) to store the pca components per head
              tensors = load_tensors_for_layer(layer_id, dataset1_path, tensor_type)

              tensors = torch.stack(tensors[:-1], dim=0)
              print (tensors.shape)
              assert num_heads == tensors.shape[-3], f"Number of heads mismatch: {num_heads} != {tensors.shape[-3]}"
              assert hidden_size == tensors.shape[-1], f"Hidden size mismatch: {hidden_size} != {tensors.shape[-1]}"

              PCA = {}

              for head in range(num_heads):
                  pca = compute_pca(layer_id, head, tensors, folder_path, tensor_type)

                  PCA[head] = pca
              # Now compute the PCA score for all the datasets for this layer and head
            
              for j, dataset2 in enumerate(datasets):
                  dataset2_path = f"{folder_path}/{dataset2}/{rtype}/{tensor_type}"

                  test_tensors = load_tensors_for_layer(layer_id, dataset2_path, tensor_type)
                  test_tensors = torch.stack(test_tensors[:-1], dim=0).cpu().numpy()
                  print (test_tensors.shape)

                  avg_score = 0
                  count = 0

                  for head in range(num_heads):
                    score = PCA[head].score(test_tensors[:,:,head,:,:].reshape(-1, test_tensors.shape[-1])[:1000, :])

                    avg_score += score
                    count += 1

                  print (f"Rotary: {rtype}, Dataset1: {dataset1}, Dataset2: {dataset2}, Layer: {layer_id}, Score: {avg_score/count}")

if __name__ == "__main__":
    main()
