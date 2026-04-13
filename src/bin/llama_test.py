import os
# Fix the OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# DO NOT change this to llama_test!
# It must stay llama_cpp to reference the installed library.
from llama_cpp import Llama

# Path to your downloaded .gguf file
model_path = r"F:\llms\gemma-4-gguf\gemma-4-26B-A4B-it-Q4_K_M.gguf"

print("[INFO] Loading Gemma 4 (26B MoE) onto 4090 VRAM...")

# Initialize the model
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, # Puts all layers on your 4090
    n_ctx=4096,      # Context length
    verbose=True     # Keep this True for now to see if CUDA is working
)

print("\n" + "="*30)
print("[SUCCESS] Model Loaded. Starting Inference...")
print("="*30)

# Run a simple completion
output = llm(
    "Q: What is the benefit of a Mixture of Experts architecture?\nA:",
    max_tokens=256,
    stop=["Q:", "\n"],
    echo=True
)

print(output["choices"][0]["text"])