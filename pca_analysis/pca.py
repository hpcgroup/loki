# This file computes the PCA of the key vectors and plots the eigenvalues
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
from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
    print (files)

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
    print (tensors.shape)

    tensors = tensors.reshape(-1, tensors.shape[-1])
    print (tensors.shape)

    # Compute the PCA of the key vectors
    pca = PCA()
    pca.fit(tensors.cpu().numpy())

    return pca.explained_variance_ratio_, pca.components_, pca.mean_

    print (pca.components_.shape)

    # Plot the eigenvalues
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Principal component index')
    plt.ylabel('Explained variance')
    plt.title('Explained variance per principal component')

    plt.show()
    plt.savefig('pca.png')
  

def main():
    # Compute the PCA of all layers and all the heads
    tensor_type = sys.argv[1] # key or query or value
    num_layers = int(sys.argv[2]) # Number of layers in the model
    folder_path = sys.argv[3] + f"/{tensor_type}" # Path to the folder containing the tensors. Except to find key, query and value subfolders in this folder
    output_dir = sys.argv[4] + f"/{tensor_type}" # Path to the output directory
    #folder_path = './topk/Mistral-7B-v0.1/1/c4/prerotary'

    
    # Initialize plotly subplot object
    #fig = make_subplots(rows=num_layers, cols=num_heads, subplot_titles=[f'Layer {i}, Head {j}' for i in range(num_layers) for j in range(num_heads)])

    # Create subdirectories for the output
    os.makedirs(f"{output_dir}/pca_components", exist_ok=True)
    os.makedirs(f"{output_dir}/pca_means", exist_ok=True)
    os.makedirs(f"{output_dir}/pca_explained_variance", exist_ok=True)
    os.makedirs(f"{output_dir}/figs", exist_ok=True)


    for layer_id in range(num_layers):
        # Tensor per layer of shape (num_heads, head_dim, head_dim) to store the pca components per head
        tensors = load_tensors_for_layer(layer_id, folder_path, tensor_type)
        tensors = torch.stack(tensors[:-1], dim=0)

        print (tensors.shape)
        num_heads = tensors.shape[-3]
        hidden_size = tensors.shape[-1]

        assert num_heads == tensors.shape[-3], f"Number of heads mismatch: {num_heads} != {tensors.shape[-3]}"
        assert hidden_size == tensors.shape[-1], f"Hidden size mismatch: {hidden_size} != {tensors.shape[-1]}"

        pca_components = torch.zeros(num_heads, hidden_size, hidden_size)
        pca_means = torch.zeros(num_heads, hidden_size)
        pca_explained_variance = torch.zeros(num_heads, hidden_size)

        for head in range(num_heads):
            explained_variance_ratio_, components, mean_ = compute_pca(layer_id, head, tensors, folder_path, tensor_type)

            # Create plotly plots for the explained variance
            #fig.add_trace(go.Scatter(x=np.arange(1, len(explained_variance_ratio_)+1), y=explained_variance_ratio_, mode='lines'), row=layer_id+1, col=head+1)

            # Add a vertical line indicating the cumulative explained variance of 95%
            #cumsum = np.cumsum(explained_variance_ratio_)
            #idx = np.argmax(cumsum >= 0.95)
            #fig.add_shape(
            #    dict(
            #        type="line",
            #        x0=idx,
            #        y0=0,
            #        x1=idx,
            #        y1=max(explained_variance_ratio_),
            #        line=dict(
            #            color="RoyalBlue",
            #            width=3,
            #        )
            #    ),
            #    row=layer_id+1, col=head+1
            #)

            # Add numpy array components to pca_components tensor
            pca_components[head] = torch.tensor(components)
            pca_means[head] = torch.tensor(mean_)
            pca_explained_variance[head] = torch.tensor(explained_variance_ratio_)

        
        torch.save(pca_components, f"{output_dir}/pca_components/pca_components_layer_{layer_id}.pt")
        torch.save(pca_means, f"{output_dir}/pca_means/pca_means_layer_{layer_id}.pt")
        torch.save(pca_explained_variance, f"{output_dir}/pca_explained_variance/pca_explained_variance_layer_{layer_id}.pt")
            

    #fig.update_layout(title_text=f"Explained variance per principal component for Llama-2-7b-hf - {tensor_type} vectors")

    # Make the figure scrollable to accomodate the 32 layers and 32 heads - really large figure
    #fig.update_layout(height=8000, width=8000, title_x=0.5)

    #fig.write_html(f"{output_dir}/figs/pca_{tensor_type}.html")

if __name__ == "__main__":
    main()
