from transformers import AutoModelForCausalLM, AutoTokenizer
#from datasets import load_dataset

#model_id = "huggyllama/llama-30b"
model_id = "mistralai/Mixtral-8x22B-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", num_proc=1)
