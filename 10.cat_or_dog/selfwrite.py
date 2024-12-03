import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import munpy as np
import time
import copy

data_dir = './data'

#数据预处理字典
data_transforms = {
        "train": transforms.Compose
        ([
            transforms.RandomRotation(45),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
    }
#数据集
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ["train", "test"]}
#print(image_datasets)
#加载器
image_dataloder = {x:torch.utils.data.DataLoader(image_datasets,batch_size=20,shuffle=True) for x in ["train", "test"]}
#print(image_dataloder)
#类别名
class_names = image_datasets["train"].classes
#print(class_names)
# 检查是否有可用的GPU并将其用于模型
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
#model = torchvision.model.to(device)

#冻结函数
def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
#迁移学习设置（借用别人的网络）
def initialize_model(model_name, num_classes, feature_extract,use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),nn.LogSoftmax(dim = 1))
        input_size = 224
    return model_ft, input_size

feature_extract = True
model_ft, input_size = initialize_model("resnet", 2, feature_extract, )#运行迁移学习初始化函数
#print(model_ft,"\n")
#print("___________ -   -   -   -   -   \n")
model_ft = model_ft.to(device)#放进gpu
#print(model_ft)
filename = "model_resnet.pkl"#为保存文件做准备

#打印要训练的(没看太懂)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

#优化器设置
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.00001)

def train_model(model, dataloder, loss_f, optimizer,num_epochs = 25,is_inception = False, filename = filename):
    since = time.time()
    best_acc = 0.0
    
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]["lr"]]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':    #训练
                model.train()
                print("training")
            else:                   #测试
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            #读取数据
            for inputs, labels in dataloder[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #清零
                optimizer.zero_grad()
                #梯度
                with torch.set_grad_enabled(phase == "train"):
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = loss_f(outputs, labels)
                        loss2 = loss_f(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = loss_f(outputs, labels)
                
                    _, pred = torch.max(outputs.data, 1)

                    #更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #计算损失
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)
        
            epoch_loss = running_loss / len(dataloder[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloder[phase].dataset)
        
            time_elapsed = time.time() - since
            print(f"Epoch {epoch+1}/{num_epochs} complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")   
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            #得到最好的模型
            if phase == "train" and epoch_acc > best_acc:
                best_acc = epoch_acc
                bast_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'test':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Best val Acc: {:4f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #用最好的当结果
    model.load_state_dict(best_model_wts)
    return model,val_acc_history,train_acc_history,valid_losses,train_losses,LRs


train_model(model_ft, image_dataloder,,criterion,num_epochs = 25,is_inception=False,filename = filename)
