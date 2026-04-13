from huggingface_hub import hf_hub_download

repo_id = "lmstudio-community/gemma-4-26B-A4B-it-GGUF"
filename = "gemma-4-26B-A4B-it-Q4_K_M.gguf"
# Exact folder from your screenshot
local_dir = r"F:\llms\gemma-4-Q4_K_M_gguf"

print(f"[INFO] Downloading the full 16GB model to {local_dir}...")
print("[NOTE] This will take a while depending on your internet speed.")

hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("[SUCCESS] Download complete. Now you can run your inference script!")