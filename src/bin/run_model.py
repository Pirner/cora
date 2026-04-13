import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Use the EXACT filename from your screenshot
gguf_file_name = "gemma-4-26B-A4B-it-Q4_K_M.gguf"
# 2. Use the EXACT folder path
model_dir = r"F:\llms\gemma-4-Q4_K_M_gguf"

print(f"[INFO] Loading local GGUF...")

model = AutoModelForCausalLM.from_pretrained(
    model_dir, # Look in this local folder for config files
    gguf_file=gguf_file_name, # Look for this file inside that folder
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Use the online repo just for the tokenizer (these are tiny files)
tokenizer = AutoTokenizer.from_pretrained("lmstudio-community/gemma-4-26B-A4B-it-GGUF")

print("\n[SUCCESS] Model is loaded on your 4090!")

prompt = "Write a short poem about a GPU."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print("\nResponse:", tokenizer.decode(outputs[0], skip_special_tokens=True))