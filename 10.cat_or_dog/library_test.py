#用来测试个别代码的

import torch
import torchvision
from torchvision import datasets, models, transforms
# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

print(torchvision.__version__)
model = models.resnet50(pretrained=True)
print(model)