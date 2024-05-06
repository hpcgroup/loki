import torch
import os

CATEGORY_TO_BASEFILE = {
  "key" : "tensor_key_",
  "query" : "tensor_query_",
  "value" : "tensor_value_",
  "attn_score" : "tensor_attn_score_",
  "attn_weights" : "tensor_attn_weights_",
}

class TensorSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.index_dict = {}
        self.last_idx = -1
        for category, basefile in CATEGORY_TO_BASEFILE.items():
            self.index_dict[category] = 0

    def save(self, category, tensor, extra_idx = None, extra_dir = ""):
        # Only save the tensor if the rank is 0
        if torch.distributed.get_rank() != 0:
            return
        # Print the first time the function is called
        if self.index_dict[category] == 0:
            print(f"Saving tensor {category} with shape {tensor.shape}")
        os.makedirs(os.path.join(self.output_dir, extra_dir, category), exist_ok=True)
        output_dir = os.path.join(self.output_dir, extra_dir, category)
        # Clear the directory if it is the first tensor
        if self.index_dict[category] == 0:
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))

        basefile = CATEGORY_TO_BASEFILE[category]
        index = self.index_dict[category]
        self.index_dict[category] += 1
        if extra_idx is not None:
            filename = f"{basefile}{extra_idx}_{index}.pt"
        else:
            filename = f"{basefile}{index}.pt"
        torch.save(tensor, os.path.join(output_dir, filename))

    def get_layer_idx(self):
        self.last_idx += 1
        return self.last_idx