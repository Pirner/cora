import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")