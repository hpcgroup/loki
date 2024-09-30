import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import torch
import numpy as np
import pandas as pd
from setup_plot import setup_global, setup_local, get_colors, get_linestyles, set_aspect_ratio, get_markers, heatmap, annotate_heatmap


setup_global()
colors = get_colors()
linestyles = get_linestyles()
markers = get_markers()

NUM_LAYERS = {
    "Llama2-7B": 32,
    "Llama2-13B": 40,
    "Llama2-70B": 80,
    "Llama3-8B": 32,
    "Llama3-70B": 80,
    "TinyLlama-1.1B": 22,
    "Mistral-7B": 32,
    "Mixtral-8x22B": 56,
    "Mixtral-8x7B": 32,
    "Pythia-6.9B": 32, 
    "Phi3-Mini-4K": 32
}

# For queries, number of heads are different for some models
NUM_HEADS = {
    "Llama2-7B": 32,
    "Llama2-13B": 40,
    "Llama2-70B": 8,
    "Llama3-8B": 8,
    "Llama3-70B": 8,
    "TinyLlama-1.1B": 4,
    "Mistral-7B": 8,
    "Mixtral-8x22B": 8,
    "Mixtral-8x7B": 8,
    "Pythia-6.9B": 32,
    "Phi3-Mini-4K":32 
}



input_dirs = {
    "Llama2-7B": {
        "wikitext": "./Llama-2-7b-hf-PCA/wikitext/",
        "c4": "./Llama-2-7b-hf-PCA/c4/",
        "bookcorpus": "./Llama-2-7b-hf-PCA/bookcorpus/",
    },
    "Llama2-13B": {
        "wikitext": "./Llama-2-13b-hf-PCA/wikitext/",
        "c4": "./Llama-2-13b-hf-PCA/c4/",
        "bookcorpus": "./Llama-2-13b-hf-PCA/bookcorpus/",
    },
    "Llama2-70B": {
        "wikitext": "./Llama-2-70b-hf-PCA/wikitext/",
        "c4": "./Llama-2-70b-hf-PCA/c4/",
        "bookcorpus": "./Llama-2-70b-hf-PCA/bookcorpus/",
    },
    "Llama3-8B": {
        "wikitext": "./Meta-Llama-3-8B-PCA/wikitext/",
        "c4": "./Meta-Llama-3-8B-PCA/c4/",
        "bookcorpus": "./Meta-Llama-3-8B-PCA/bookcorpus/",
    },
    "Llama3-70B": {
        "wikitext": "./Meta-Llama-3-70B-PCA/wikitext/",
        "c4": "./Meta-Llama-3-70B-PCA/c4/",
        "bookcorpus": "./Meta-Llama-3-70B-PCA/bookcorpus/",
    },
    "TinyLlama-1.1B": {
        "wikitext": "./TinyLlama-1.1B-Chat-v1.0-PCA/wikitext/",
        "c4": "./TinyLlama-1.1B-Chat-v1.0-PCA/c4/",
        "bookcorpus": "./TinyLlama-1.1B-Chat-v1.0-PCA/bookcorpus/",
    },
    "Mistral-7B": {
        "wikitext": "./Mistral-7B-v0.1-PCA/wikitext/",
        "c4": "./Mistral-7B-v0.1-PCA/c4/",
        "bookcorpus": "./Mistral-7B-v0.1-PCA/bookcorpus/"
    },
    "Mixtral-8x7B": {
        "wikitext": "./Mixtral-8x7B-v0.1-PCA/wikitext/",
        "c4": "./Mixtral-8x7B-v0.1-PCA/bookcorpus/",
        "bookcorpus": "./Mixtral-8x7B-v0.1-PCA/bookcorpus/"
    },
    "Mixtral-8x22B": {
        "wikitext": "./Mixtral-8x22B-v0.1-PCA/wikitext/",
        "bookcorpus": "./Mixtral-8x22B-v0.1-PCA/bookcorpus/"
    },
    "Pythia-6.9B": {
        "wikitext": "./pythia-6.9b-PCA/wikitext/",
        "c4": "./pythia-6.9b-PCA/c4/",
        "bookcorpus": "./pythia-6.9b-PCA/bookcorpus/"
    },
    #"Phi3-Mini-4K": {
    #    "wikitext": "./Phi3-mini-4k-PCA/wikitext/",
    #    "c4": "./Phi3-mini-4k-PCA/c4/",
    #    "bookcorpus": "./Phi3-mini-4k-PCA/bookcorpus/",
    #}
}

ranks_at_95 = {
    "Llama2-7B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "Llama2-13B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "Llama2-70B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "Llama3-8B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "Llama3-70B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "TinyLlama-1.1B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "Mistral-7B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": [],
    },
    "Mixtral-8x7B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": [],
    },
    "Mixtral-8x22B": {
      "wikitext": [],
      "bookcorpus": [],
    },
    "Pythia-6.9B": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": []
    },
    "Phi3-Mini-4K": {
      "wikitext": [],
      "c4": [],
      "bookcorpus": [],
    }
}

dataset_name_dict = {
    "wikitext": "Wikitext",
    "c4": "C4",
    "bookcorpus": "BookCorpus"
}

rotary_type_name_dict = {
    "postrotary": "Post-Rotary",
    "prerotary": "Pre-Rotary"
}


def get_pca_components (layer_id, folder_path, tensor_type='key'):
    data_folder = os.path.join(folder_path, tensor_type)
    pca_components = torch.load(f"{data_folder}/pca_components/pca_components_layer_{layer_id}.pt")
    return pca_components

def get_pca_data(layer_id, folder_path, tensor_type='key'):
    data_folder = os.path.join(folder_path, tensor_type)
    explained_variance = torch.load(f"{data_folder}/pca_explained_variance/pca_explained_variance_layer_{layer_id}.pt")
    return explained_variance

def plot_rank_at_k(output_dir, tensor_type, rotary_type):
    output_dir = os.path.join(output_dir, tensor_type, rotary_type)
    os.makedirs(output_dir, exist_ok=True)

    for model_name in input_dirs.keys():
        os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)

    for model_name, input_dict in input_dirs.items():
        i= 0
        setup_local()
        num_layers = NUM_LAYERS[model_name]
        expected_num_heads = NUM_HEADS[model_name]
        hidden_size = 128
        for dataset, input_dir in input_dict.items():
            input_dir = os.path.join(input_dir, rotary_type)
            avg_ranks_at_95 = []
            stddev_ranks_at_95 = []
            for layer_id in range(num_layers):
                # Tensor per layer of shape (num_heads, head_dim, head_dim) to
                # store the pca components per head
                variances = get_pca_data(layer_id, input_dir, tensor_type)
                num_heads = variances.shape[0]
                hidden_size = variances.shape[-1]

                assert num_heads == expected_num_heads, f"{model_name}: Expected {expected_num_heads} heads, got {num_heads}"

                rank_at_95 = np.argmax(np.cumsum(variances, axis=1) > 0.90, axis=1).to(torch.float32)
                ranks_at_95[model_name][dataset].append(rank_at_95)
                # Find the average rank across all heads
                avg_ranks_at_95.append(rank_at_95.mean().item())
                stddev_ranks_at_95.append(rank_at_95.std().item())

            plt.plot(range(len(avg_ranks_at_95)), avg_ranks_at_95, label=dataset_name_dict[dataset], 
                     color=colors[i], linestyle=linestyles[i], marker=markers[i])
            # Add a dashed line at the average rank across all layers
            #plt.axhline(y=np.mean(avg_ranks_at_95), color=colors[i],
            #linestyle='dashed')

            # Plot the line plot with error bars
            #plt.errorbar(range(len(avg_ranks_at_95)), avg_ranks_at_95,
            #yerr=stddev_ranks_at_95, label=dataset, color=colors[i],
            #linestyle=linestyles[i], marker=markers[i])

            i = i + 1
        # Add a horizontal line at 128
        if model_name == "Phi3-Mini-4K":
            plt.axhline(y=96, color=colors[i+1], linestyle='dashed')
            yticks = [0, 20, 40, 60, 80, 96, 100]
        else:
            plt.axhline(y=hidden_size, color=colors[i+1], linestyle='dashed')
            yticks = [0, 20, 40, 60, 80, 100, 120, 128, 140]
        if tensor_type == "query":
            plt.title(f"{model_name} Attention Queries ({rotary_type_name_dict[rotary_type]})", fontsize=22)
        else:
            plt.title(f"{model_name} Attention {tensor_type.capitalize()}s ({rotary_type_name_dict[rotary_type]})", fontsize=22)
        plt.ylabel("Rank at 90% Explained Variance", fontsize=22)
        plt.xlabel("Layer", fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 140)
        plt.xlim(0, num_layers)
        plt.yticks(yticks)
        plt.legend(loc='best', fontsize=20) 
        plt.savefig(f"{output_dir}/{model_name}/{model_name}_avg_ranks_at_90.pdf", bbox_inches='tight')

def plot_pca_components(output_dir, tensor_type, rotary_type):
    output_dir = os.path.join(output_dir, tensor_type)

    # For certain layer and head, plot the explained variance of the PCA components
    sample_layer_id = [1, 21]
    sample_head_id = [0, 3]
    for model_name, input_dict in input_dirs.items():
        num_layers = NUM_LAYERS[model_name]
        for j, layer_id in enumerate(sample_layer_id):
            i= 0
            setup_local()

            dataset = "wikitext"
            input_dir = input_dict[dataset]
            print (f"Plotting for {model_name}, Layer {layer_id}, Head {sample_head_id[j]}")
            print (f"Input dir: {input_dir}")
            for rotary in rotary_type:
                input_rotary_dir = os.path.join(input_dir, rotary)
                variances = get_pca_data(layer_id, input_rotary_dir, tensor_type)

                plt.plot(range(variances.shape[1]), variances[sample_head_id[j]].numpy(), 
                         label=f"{rotary_type_name_dict[rotary]}", color=colors[i], linestyle="solid")
                rank_at_95 = np.argmax(np.cumsum(variances[sample_head_id[j]]) > 0.90).item()
                # Add a vertical line at the rank at 95% explained variance
                plt.axvline(x=rank_at_95, color=colors[i], linestyle='dashed')
                i = i + 1

            plt.title(f"{model_name}, Layer {layer_id}, Head {sample_head_id[j]}", fontsize=22)
            plt.ylabel("Normalised Eigen-values", fontsize=20)
            plt.xlabel("Feature Dimension / Rank", fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim(0, 0.3)
            plt.xlim(0, 128)
            plt.legend(loc='best')
            plt.savefig(f"{output_dir}/{model_name}_{layer_id}_variances.pdf", bbox_inches='tight')

def plot_heatmap(output_dir, tensor_type, rotary_type):
    output_dir = os.path.join(output_dir, tensor_type)
    os.makedirs(output_dir, exist_ok=True)

    fig_shapes = { 
        "Llama2-7B": (20, 10), 
        "Mistral-7B": (20, 5),
    }

    for model_name, input_dict in input_dirs.items():
        i= 0
        dataset = "wikitext"
        input_dir = input_dict[dataset]
        num_layers = NUM_LAYERS[model_name]
        for rotary in rotary_type:
            output_rotary_dir = os.path.join(output_dir, rotary)
            input_rotary_dir = os.path.join(input_dir, rotary)
            ranks_at_95[model_name][dataset] = []
            for layer_id in range(num_layers):
                variances = get_pca_data(layer_id, input_rotary_dir, tensor_type)

                rank_at_95 = np.argmax(np.cumsum(variances, axis=1) > 0.90, axis=1).to(torch.float32)
                ranks_at_95[model_name][dataset].append(rank_at_95)


            ranks_at_95_tensor = torch.stack(ranks_at_95[model_name][dataset]).transpose(0, 1) 
            layers = range(ranks_at_95_tensor.shape[1]) 
            heads = range(ranks_at_95_tensor.shape[0])

            setup_local()
            fig, ax = plt.subplots(figsize=fig_shapes[model_name]) 
            im, cbar = heatmap(ranks_at_95_tensor, heads, layers, ax=ax, cmap="Blues",
                               cbarlabel="Rank at 90%") 
            if model_name == "Mistral-7B": 
              annotate_heatmap(im, valfmt="{x:.0f}", size = 11) 
            elif model_name == "Llama2-7B": annotate_heatmap(im, valfmt="{x:.0f}", size = 11)

            fig.tight_layout()
            plt.xlabel("Layers")
            plt.ylabel("Heads")
            plt.savefig(f"{output_rotary_dir}/{model_name}/ranks_at_90_heatmap.pdf", bbox_inches='tight')
            plt.close(fig)


def compare_pca_components(output_dir, tensor_type, rotary_type):
    # We want to compare the PCA components of the same layer and head for a model computed across 
    # different datasets
    output_dir = os.path.join(output_dir, tensor_type)
            
    for model_name, input_dict in input_dirs.items():
        num_layers = NUM_LAYERS[model_name]
        for layer_id in range(num_layers):
            for rotary in rotary_type:
                components_dict = {}
                variance_dict = {}
                for dataset, input_dir in input_dict.items():
                    input_rotary_dir = os.path.join(input_dir, rotary)
                    components = get_pca_components(layer_id, input_rotary_dir, tensor_type)
                    components_dict[dataset] = components
                    variance_dict[dataset] = get_pca_data(layer_id, input_rotary_dir, tensor_type)
                # Compute the cosine similarity between the PCA components of the same layer and head
                # across different datasets and compute a grid of cosine similarities between the datasets
                sims = {}
                for i, dataset1 in enumerate(input_dict.keys()):
                    sims[dataset1] = {}
                    for j, dataset2 in enumerate(input_dict.keys()):
                        if i != j:
                            components1 = components_dict[dataset1]
                            components2 = components_dict[dataset2]
                            variances1 = variance_dict[dataset1]
                            variances2 = variance_dict[dataset2]
                            #print (variances1.shape)
                            #print (variances1[0, :])
                            #print (variances2[0, :])
                            cos_sim = torch.nn.functional.cosine_similarity(components1, components2, dim=-1)
                            cos_sim = cos_sim[:, :20]
                            cos_sim = torch.abs(cos_sim)
                            cos_sim = cos_sim.mean()


                            l2_norm = torch.norm(variances1 - variances2, dim=-1).mean()

                            sims[dataset1][dataset2] = cos_sim.item()
                
                print (sims)



def plot_pca_rank_across_models(output_dir, tensor_type, rotary_type):
    output_dir = os.path.join(output_dir, tensor_type)

    setup_local()
    dataset = "wikitext"

    # Plot all models except for Phi3-Mini-4K and TinyLlama-1.1B
    input_dirs_local = input_dirs.copy()
    del input_dirs_local["Phi3-Mini-4K"]
    del input_dirs_local["TinyLlama-1.1B"]


    if tensor_type == "query":
        plt.title(f"Dimensionality of Attention Queries")
    else:
        plt.title(f"Dimensionality of Attention {tensor_type.capitalize()}s")
    plt.ylabel("Rank at 90% Explained Variance")
    plt.xlabel("Model")
    plt.xticks(range(len(input_dirs_local.keys())), input_dirs_local.keys(), rotation=30)
    # Rotate the x-axis labels
    # Remove the x-axis
    #plt.gca().axes.get_xaxis().set_visible(False)
    plt.axhline(y=128, color="black", linestyle='dashed')
    yticks = [0, 20, 40, 60, 80, 100, 120, 128, 140]
    plt.ylim(0, 140)
    plt.yticks(yticks)
    # For certain layer and head, plot the explained variance of the PCA components
    i= 0
    for rotary in ['postrotary']:
        for model_name, input_dict in input_dirs_local.items():
            num_layers = NUM_LAYERS[model_name]
            input_dir = input_dict[dataset]
            avg_ranks_at_95 = []
            stddev_ranks_at_95 = []
            for layer_id in range(num_layers):
                input_rotary_dir = os.path.join(input_dir, rotary)
                variances = get_pca_data(layer_id, input_rotary_dir, tensor_type)

                rank_at_95 = np.argmax(np.cumsum(variances, axis=1) > 0.90, axis=1).to(torch.float32)

                # Find the average rank across all heads
                avg_ranks_at_95.append(rank_at_95.mean().item())
                stddev_ranks_at_95.append(rank_at_95)
            
            avg_rank_at_95 = np.mean(avg_ranks_at_95)

            stddev_ranks_at_95 = np.stack(stddev_ranks_at_95).reshape(-1)
            stddev_rank_at_95 = np.std(stddev_ranks_at_95)

            # For this model, plot the average rank at 90% explained variance
            # and the standard deviation around that 1 point on the plot
            color_index = int(i % len(colors))
            linestyle_index = int(i / len(linestyles))
            marker_index = int(i % len(markers))
            errorbars = plt.errorbar(i, avg_rank_at_95, yerr=stddev_rank_at_95, capsize=5, label=model_name,
                         color=colors[color_index], fmt=markers[marker_index], 
                         markersize=7)
                  
            for err_bar in errorbars[2]:
                err_bar.set_linestyle('--')


            # Make the legend outside the plot below the x-axis
            #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)
            #plt.legend(loc='best', ncol=3)
            # Remove the x-axis label
            plt.xlabel('')
            plt.savefig(f"{output_dir}/global_{rotary}_variances.pdf", bbox_inches='tight')
            i = i + 1

                

def main():
    tensor_type = 'key'
    rotary_type = ['prerotary', 'postrotary']
    output_dir = sys.argv[1]

    for rotary in rotary_type:
        plot_rank_at_k(output_dir, tensor_type, rotary)
    
    plot_pca_components(output_dir, tensor_type, rotary_type)

    plot_pca_rank_across_models(output_dir, tensor_type, rotary_type)

    plot_heatmap(output_dir, tensor_type, rotary_type)

    #compare_pca_components(output_dir, tensor_type, rotary_type)

if __name__ == "__main__":
    main()
