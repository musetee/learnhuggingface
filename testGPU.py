# check if GPU is available
import torch
import os
print(torch.__version__)

print(torch.version.cuda)  
print(torch.backends.cudnn.version())

print(torch.cuda.is_available())