from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_id = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Enable this to cache the dataset
#test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", num_proc=1)
