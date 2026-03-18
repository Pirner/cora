from huggingface_hub import snapshot_download

# This is the model you want
model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
model_id = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit"

print(f"Starting download for {model_id}...")
print("Note: This is a ~45GB download. Please ensure you have enough space on F:")

snapshot_download(
    repo_id=model_id,
    local_dir="F:/llms/mistral_ai_small_3.2_24b_instruct",
    local_dir_use_symlinks=False,  # Important for Windows stability
    ignore_patterns=["original/*"], # Replaces your --exclude flag
    resume_download=True           # If your internet cuts out, it will pick up where it left off
)

print("Download complete!")