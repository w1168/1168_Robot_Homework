import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image

def load_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #image_datasets = {datasets.ImageFolder(root="./single_test",transform = transform)}
    #date_loader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True)
    image = Image.open("./single_test/2140e36c-f8a4-48ca-9fca-5286703c86f5.png")
    image = transform(image).unsqueeze(0)
    return image

def use_model(model, image , device):
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        inputs = image.to(device)
        #labels = labels.to(device)
        labels_pred = model(inputs)
        print(labels_pred)
        _,pred = torch.max(labels_pred.data, 1)
        print(pred)
        if pred[0].item() == 0:
            print("Prediction:", "cat")
        else:
            print("Prediction:", "dog")





if __name__ == "__main__":
    print("Start")
# 加载预训练的ResNet-50模型并修改最后一层(false)
    model = models.resnet50()
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(2048, 2)

# 加载数据集和创建数据加载器
    data_loader = load_data()
    #print(data_loader)
# 定义损失函数和优化器
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.00001)

#检查是否有可用的GPU并将其用于模型
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

#加载模型参数
    model.load_state_dict(torch.load("./models/model_epoch_3.pth", map_location=device))

    use_model(model, data_loader,device)