import sys, os
import imageio
print("sys pathes:", sys.path)

from sysconfig import get_paths
from pprint import pprint

info = get_paths()  # a dictionary of key-paths

# pretty print it for now
pprint(info)

import numpy as np
print("numpy version:", np.version.version)
print("numpy path:", np.get_include())

import torch

print("torch version:", torch.__version__)
withCuda = torch.cuda.is_available()
if withCuda:
    print("CUDA version:", torch.version.cuda)
    cuda_N = torch.cuda.device_count()
    print("GPU count:", cuda_N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda_N > 0:
        print("GPU name:",torch.cuda.get_device_name(device))
        print("GPU properties:", torch.cuda.get_device_properties(device))
        print("GPU total memory:", torch.cuda.get_device_properties(device).total_memory)
    print("The code was tested on a single NVIDIA Quadro RTX 8000 GPU with 48G memory.\nTo work with smaller memory, parameters need modifications.\nTo use multiple GPUs, code needs to be modified.")