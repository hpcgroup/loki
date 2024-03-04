import torch
import os

CATEGORY_TO_BASEFILE = {
  "keys" : "tensor_key_",
  "query" : "tensor_query_",
  "attn_score" : "tensor_attn_score_",
  "attn_weights" : "tensor_attn_weights_",
}

class TensorSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.index_dict = {}
        for category, basefile in CATEGORY_TO_BASEFILE.items():
            self.index_dict[category] = 0

    def save(self, category, tensor, extra_idx = None):
        basefile = CATEGORY_TO_BASEFILE[category]
        index = self.index_dict[category]
        self.index_dict[category] += 1
        if extra_idx is not None:
            filename = f"{basefile}{extra_idx}_{index}.pt"
        else:
            filename = f"{basefile}{index}.pt"
        torch.save(tensor, os.path.join(self.output_dir, filename))